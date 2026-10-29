//! Document reads, partial updates, and deletes.

use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use reqwest::StatusCode;
use reqwest::blocking::{RequestBuilder, Response};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;

use super::{NightscoutClient, url_encode_query, value_kind};

/// How [`NightscoutClient::delete_document`] removes a document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeleteMode {
    /// Marks the document `isValid: false`. v3 reads skip it, but the v1 API
    /// and the web UI still show it.
    Soft,
    /// Removes the document from the database.
    Permanent,
}

impl NightscoutClient {
    /// GET `api/v3/{collection}?{query}` and decode each document in the result.
    ///
    /// `query` is sent as-is, so percent-encode any values that need it.
    pub fn search<T: DeserializeOwned>(&self, collection: &str, query: &str) -> Result<Vec<T>> {
        let url = format!("{}api/v3/{collection}?{query}", self.base_url);
        let resp = self
            .send_with_reauth(|bearer| self.http.get(&url).bearer_auth(bearer))
            .with_context(|| format!("GET {url}"))?;
        let body: Value = require_success(resp, "GET", &url)?
            .json()
            .with_context(|| format!("parsing Nightscout search response from {url}"))?;
        search_results(body)
            .with_context(|| format!("parsing Nightscout search response from {url}"))?
            .into_iter()
            .map(serde_json::from_value)
            .collect::<Result<Vec<T>, _>>()
            .with_context(|| format!("decoding Nightscout documents from {url}"))
    }

    /// PATCH `api/v3/{collection}/{identifier}` with the given fields.
    ///
    /// Unlike re-POSTing a document, this can change one that another app
    /// created: Nightscout refuses a re-POST that would change `app`.
    pub fn patch_document<T: Serialize>(
        &self,
        collection: &str,
        identifier: &str,
        fields: &T,
    ) -> Result<()> {
        let url = self.document_url(collection, identifier);
        let resp = self
            .send_with_reauth(|bearer| self.http.patch(&url).bearer_auth(bearer).json(fields))
            .with_context(|| format!("PATCH {url}"))?;
        require_success(resp, "PATCH", &url).map(drop)
    }

    /// DELETE `api/v3/{collection}/{identifier}`. A document that is already
    /// gone (404 or 410) counts as deleted.
    pub fn delete_document(
        &self,
        collection: &str,
        identifier: &str,
        mode: DeleteMode,
    ) -> Result<()> {
        let mut url = self.document_url(collection, identifier);
        if mode == DeleteMode::Permanent {
            // Nightscout recognizes exactly `permanent`; any other name falls
            // back to a soft delete and still returns 200.
            url.push_str("?permanent=true");
        }
        let resp = self
            .send_with_reauth(|bearer| self.http.delete(&url).bearer_auth(bearer))
            .with_context(|| format!("DELETE {url}"))?;
        if matches!(resp.status(), StatusCode::NOT_FOUND | StatusCode::GONE) {
            return Ok(());
        }
        require_success(resp, "DELETE", &url).map(drop)
    }

    fn document_url(&self, collection: &str, identifier: &str) -> String {
        format!(
            "{}api/v3/{collection}/{}",
            self.base_url,
            url_encode_query(identifier)
        )
    }

    /// Send the request `build` makes, re-authenticating and sending it again
    /// once after a 401.
    fn send_with_reauth(&self, build: impl Fn(&str) -> RequestBuilder) -> Result<Response> {
        let bearer_first = self
            .bearer()
            .ok_or_else(|| anyhow!("NightscoutClient.authenticate() must be called first"))?;
        let resp = send(build(&bearer_first))?;
        if resp.status() != StatusCode::UNAUTHORIZED {
            return Ok(resp);
        }
        let bearer_retry = self.refresh_bearer_after_401(&bearer_first)?;
        send(build(&bearer_retry))
    }
}

fn send(request: RequestBuilder) -> Result<Response> {
    request
        .header("Accept", "application/json")
        .timeout(Duration::from_secs(60))
        .send()
        .context("request failed to send")
}

fn require_success(resp: Response, method: &str, url: &str) -> Result<Response> {
    let status = resp.status();
    if status.is_success() {
        return Ok(resp);
    }
    let body = resp
        .text()
        .unwrap_or_else(|_| "<failed to read response body>".into());
    bail!("Nightscout {method} {url} returned {status}: {body}")
}

/// The documents in a search response, either a v3 `{"result": [...]}`
/// envelope or a bare array.
fn search_results(body: Value) -> Result<Vec<Value>> {
    match body {
        Value::Array(items) => Ok(items),
        Value::Object(mut map) => match map.remove("result") {
            Some(Value::Array(items)) => Ok(items),
            Some(Value::Null) => Ok(Vec::new()),
            Some(other) => bail!(
                "Nightscout search response `result` was {}, expected array",
                value_kind(&other)
            ),
            None => bail!("Nightscout search response object missing `result`"),
        },
        other => bail!(
            "Nightscout search response was {}, expected array or object",
            value_kind(&other)
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal::Decimal;
    use rust_decimal_macros::dec;
    use serde::Deserialize;

    #[derive(Debug, Deserialize)]
    struct Meal {
        identifier: Option<String>,
        #[serde(default, with = "rust_decimal::serde::arbitrary_precision_option")]
        carbs: Option<Decimal>,
        #[serde(default, with = "rust_decimal::serde::arbitrary_precision_option")]
        duration: Option<Decimal>,
    }

    #[test]
    fn search_results_unwraps_the_v3_envelope() {
        let body = serde_json::json!({"status": 200, "result": [{"identifier": "meal-1"}]});
        assert_eq!(search_results(body).unwrap().len(), 1);

        let null = serde_json::json!({"status": 200, "result": null});
        assert!(search_results(null).unwrap().is_empty());
    }

    #[test]
    fn search_results_accepts_a_bare_array() {
        let body = serde_json::json!([{"identifier": "meal-1"}, {"identifier": "meal-2"}]);
        assert_eq!(search_results(body).unwrap().len(), 2);
    }

    #[test]
    fn search_results_rejects_an_unexpected_envelope() {
        let body = serde_json::json!({"status": 200, "result": {"count": 0}});
        assert!(search_results(body).is_err());
        assert!(search_results(serde_json::json!("nope")).is_err());
    }

    #[test]
    fn search_results_decode_decimal_fields_from_numbers() {
        let body: Value = serde_json::from_str(
            r#"{"result": [{"identifier": "meal-1", "carbs": 58, "duration": 180.5}, {"identifier": "meal-2"}]}"#,
        )
        .unwrap();
        let meals: Vec<Meal> = search_results(body)
            .unwrap()
            .into_iter()
            .map(serde_json::from_value)
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(meals[0].identifier.as_deref(), Some("meal-1"));
        assert_eq!(meals[0].carbs, Some(dec!(58)));
        assert_eq!(meals[0].duration, Some(dec!(180.5)));
        assert_eq!(meals[1].carbs, None);
    }

    #[test]
    fn document_url_encodes_the_identifier() {
        let ns = NightscoutClient::new("https://ns.example".into(), "role".into());
        assert_eq!(
            ns.document_url("treatments", "meal 1/2"),
            "https://ns.example/api/v3/treatments/meal%201%2F2"
        );
    }
}
