//! Nightscout v3 HTTP client.

use std::sync::{Mutex, RwLock};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use reqwest::StatusCode;
use reqwest::blocking::Client;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use secrecy::{ExposeSecret, SecretString};
use serde::Serialize;
use serde_json::Value;

use crate::models::NightscoutBearer;

mod documents;
pub use documents::DeleteMode;

// TODO: Don't mix const types.

/// Bolus dedup window.
pub const BOLUS_DEDUP_WINDOW_MS: i64 = 120_000;

/// Non-bolus treatment dedup window.
pub const TREATMENT_DEDUP_WINDOW_MS: i64 = 120_000;

/// Insulin tolerance for [`NightscoutClient::has_matching_bolus`].
pub const BOLUS_DEDUP_EPSILON: Decimal = dec!(0.009);

pub struct NightscoutClient {
    pub base_url: String,
    pub permission_role: String,
    pub http: Client,
    /// Shared across rayon POSTs and refreshed in place on 401.
    bearer: RwLock<Option<SecretString>>,
    reauth_guard: Mutex<()>,
}

impl NightscoutClient {
    pub fn new(base_url: String, permission_role: String) -> Self {
        let base_url = if base_url.ends_with('/') {
            base_url
        } else {
            format!("{base_url}/")
        };
        Self {
            base_url,
            permission_role,
            http: Client::new(),
            bearer: RwLock::new(None),
            reauth_guard: Mutex::new(()),
        }
    }

    pub fn with_http(mut self, http: Client) -> Self {
        self.http = http;
        self
    }

    /// Fetch and store a Nightscout bearer token.
    pub fn authenticate(&self) -> Result<()> {
        let url = format!(
            "{}api/v2/authorization/request/{}",
            self.base_url, self.permission_role
        );
        let bearer: NightscoutBearer = self
            .http
            .get(&url)
            .send()
            .with_context(|| format!("Nightscout token request failed ({url})"))?
            .error_for_status()
            .context("Nightscout token request returned non-2xx")?
            .json()
            .context("Nightscout token response was not the expected shape")?;
        let mut guard = self.bearer.write().unwrap_or_else(|p| p.into_inner());
        *guard = Some(SecretString::new(bearer.token.into()));
        Ok(())
    }

    pub(crate) fn bearer(&self) -> Option<String> {
        self.bearer
            .read()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
            .map(|s| s.expose_secret().to_owned())
    }

    /// POST: Retries once after a 401. 400 means
    /// dedup hit, and is treated as success.
    pub fn post_document<T: Serialize>(&self, collection: &str, doc: &T) -> Result<()> {
        let url = format!("{}api/v3/{}", self.base_url, collection);

        let bearer_first = self
            .bearer()
            .ok_or_else(|| anyhow!("NightscoutClient.authenticate() must be called first"))?;
        match self.post_once(&url, &bearer_first, doc)? {
            PostOutcome::Ok | PostOutcome::Dedup => return Ok(()),
            PostOutcome::HttpError { status, body } if status != StatusCode::UNAUTHORIZED => {
                return Err(anyhow!("Nightscout POST {url} returned {status}: {body}"));
            }
            PostOutcome::HttpError { .. } => {
                // 401: re-auth and retry once.
            }
        }

        let bearer_retry = self.refresh_bearer_after_401(&bearer_first)?;
        match self.post_once(&url, &bearer_retry, doc)? {
            PostOutcome::Ok | PostOutcome::Dedup => Ok(()),
            PostOutcome::HttpError { status, body } => Err(anyhow!(
                "Nightscout POST {url} returned {status} after bearer refresh: {body}"
            )),
        }
    }

    /// Check for a nearby bolus with matching insulin amount.
    ///
    /// This catches cross-tool duplicates whose timestamps differ.
    pub fn has_matching_bolus(
        &self,
        date_ms: i64,
        insulin: Decimal,
        window_ms: i64,
        epsilon: Decimal,
    ) -> Result<bool> {
        let url = format!(
            "{}api/v3/treatments?date$gte={}&date$lte={}&insulin$gte={}&insulin$lte={}&fields=identifier,date,insulin&limit=1",
            self.base_url,
            date_ms - window_ms,
            date_ms + window_ms,
            insulin - epsilon,
            insulin + epsilon,
        );
        self.search_treatments_with_retry(&url)
    }

    /// Check for a nearby treatment of a specific event type.
    ///
    /// Catches cross-tool duplicates.
    pub fn has_matching_treatment(
        &self,
        event_type: &str,
        date_ms: i64,
        window_ms: i64,
    ) -> Result<bool> {
        let event_type_enc = url_encode_query(event_type);
        let url = format!(
            "{}api/v3/treatments?eventType={event_type_enc}&date$gte={}&date$lte={}&fields=identifier,date,eventType&limit=1",
            self.base_url,
            date_ms - window_ms,
            date_ms + window_ms,
        );
        self.search_treatments_with_retry(&url)
    }

    /// GET treatments, refreshing the bearer once after a 401.
    fn search_treatments_with_retry(&self, url: &str) -> Result<bool> {
        let bearer_first = self
            .bearer()
            .ok_or_else(|| anyhow!("NightscoutClient.authenticate() must be called first"))?;
        if let Some(t) = self.search_once(url, &bearer_first)? {
            return Ok(t);
        }
        let bearer_retry = self.refresh_bearer_after_401(&bearer_first)?;
        match self.search_once(url, &bearer_retry)? {
            Some(t) => Ok(t),
            None => bail!("Nightscout GET {url} returned 401 even after bearer refresh"),
        }
    }

    /// Serialize bearer refreshes after a 401.
    fn refresh_bearer_after_401(&self, observed: &str) -> Result<String> {
        {
            let _guard = self.reauth_guard.lock().unwrap_or_else(|p| p.into_inner());
            let bearer_now = self
                .bearer()
                .ok_or_else(|| anyhow!("bearer cleared under reauth guard"))?;
            if bearer_now == observed {
                self.authenticate()
                    .context("refreshing Nightscout bearer after 401")?;
            }
        }
        self.bearer()
            .ok_or_else(|| anyhow!("bearer missing after refresh"))
    }

    fn search_once(&self, url: &str, bearer: &str) -> Result<Option<bool>> {
        let resp = self
            .http
            .get(url)
            .header("Accept", "application/json")
            .bearer_auth(bearer)
            .timeout(Duration::from_secs(60))
            .send()
            .with_context(|| format!("GET {url} failed to send"))?;

        if resp.status() == StatusCode::UNAUTHORIZED {
            return Ok(None);
        }
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp
                .text()
                .unwrap_or_else(|_| "<failed to read response body>".into());
            bail!("Nightscout GET {url} returned {status}: {body}");
        }
        let body: Value = resp
            .json()
            .with_context(|| format!("parsing Nightscout search response from {url}"))?;
        response_has_results(&body)
            .with_context(|| format!("parsing Nightscout search response from {url}"))
            .map(Some)
    }

    fn post_once<T: Serialize>(&self, url: &str, bearer: &str, doc: &T) -> Result<PostOutcome> {
        let resp = self
            .http
            .post(url)
            .header("Accept", "application/json")
            .header("Content-Type", "application/json")
            .bearer_auth(bearer)
            .timeout(Duration::from_secs(60))
            .json(doc)
            .send()
            .with_context(|| format!("POST {url} failed to send"))?;

        if resp.status().is_success() {
            return Ok(PostOutcome::Ok);
        }

        let status = resp.status();
        let body = resp
            .text()
            .unwrap_or_else(|_| "<failed to read response body>".into());

        if status == StatusCode::BAD_REQUEST && is_immutable_field_dedup(&body) {
            return Ok(PostOutcome::Dedup);
        }

        Ok(PostOutcome::HttpError { status, body })
    }
}

/// Detect Nightscout's "this identifier exists, with conflicting immutable
/// fields" 400 response. Treated as a successful dedup.
fn is_immutable_field_dedup(body: &str) -> bool {
    if let Ok(v) = serde_json::from_str::<Value>(body)
        && let Some(msg) = v
            .get("message")
            .or_else(|| v.get("error"))
            .and_then(|m| m.as_str())
        && msg
            .to_ascii_lowercase()
            .contains("cannot be modified by the client")
    {
        return true;
    }
    body.to_ascii_lowercase()
        .contains("cannot be modified by the client")
}

fn response_has_results(body: &Value) -> Result<bool> {
    match body {
        Value::Array(items) => Ok(!items.is_empty()),
        Value::Object(map) => match map.get("result") {
            Some(Value::Array(items)) => Ok(!items.is_empty()),
            Some(Value::Null) => Ok(false),
            Some(other) => bail!(
                "Nightscout search response `result` was {}, expected array",
                value_kind(other)
            ),
            None => bail!("Nightscout search response object missing `result`"),
        },
        other => bail!(
            "Nightscout search response was {}, expected array or object",
            value_kind(other)
        ),
    }
}

fn value_kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Percent-encode an event type for use in a NS query string.
fn url_encode_query(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => out.push(c),
            _ => {
                let mut buf = [0u8; 4];
                for b in c.encode_utf8(&mut buf).bytes() {
                    out.push_str(&format!("%{b:02X}"));
                }
            }
        }
    }
    out
}

enum PostOutcome {
    Ok,
    Dedup,
    HttpError { status: StatusCode, body: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn immutable_field_dedup_matches_legacy_substring() {
        // Older NS versions return a free-form string body.
        let legacy =
            "Treatment with identifier exists; field `app` cannot be modified by the client";
        assert!(is_immutable_field_dedup(legacy));
    }

    #[test]
    fn immutable_field_dedup_matches_structured_message() {
        // Newer NS returns a structured JSON body.
        let body = r#"{"status":400,"message":"cannot be modified by the client"}"#;
        assert!(is_immutable_field_dedup(body));
    }

    #[test]
    fn immutable_field_dedup_is_case_insensitive() {
        let body = "Cannot Be Modified By The Client";
        assert!(is_immutable_field_dedup(body));
    }

    #[test]
    fn immutable_field_dedup_rejects_unrelated_400() {
        let body = r#"{"status":400,"message":"missing required field `date`"}"#;
        assert!(!is_immutable_field_dedup(body));
    }

    #[test]
    fn search_response_accepts_bare_array() {
        let body = serde_json::json!([{"identifier": "bolus-1"}]);
        assert!(response_has_results(&body).unwrap());

        let empty = serde_json::json!([]);
        assert!(!response_has_results(&empty).unwrap());
    }

    #[test]
    fn search_response_accepts_v3_result_envelope() {
        let body = serde_json::json!({
            "status": 200,
            "result": [{"identifier": "bolus-1"}]
        });
        assert!(response_has_results(&body).unwrap());

        let empty = serde_json::json!({"status": 200, "result": []});
        assert!(!response_has_results(&empty).unwrap());
    }

    #[test]
    fn search_response_rejects_unexpected_envelope() {
        let body = serde_json::json!({"status": 200, "result": {"count": 0}});
        assert!(response_has_results(&body).is_err());
    }

    #[test]
    fn url_encode_query_handles_spaces_and_punct() {
        assert_eq!(url_encode_query("Site Change"), "Site%20Change");
        assert_eq!(url_encode_query("Suspend Pump"), "Suspend%20Pump");
        // No encoding for unreserved chars.
        assert_eq!(url_encode_query("abc-_.~"), "abc-_.~");
    }
}
