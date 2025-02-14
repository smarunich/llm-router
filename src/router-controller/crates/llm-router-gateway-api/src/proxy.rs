//! Proxy
use crate::config::{Policy, RouterConfig};
use crate::error::GatewayApiError;
use crate::metrics::{
    track_token_usage, LLM_RESPONSE_TIME, MODEL_SELECTION_TIME, NUM_REQUESTS,
    PROXY_OVERHEAD_LATENCY, REQUESTS_PER_MODEL, REQUESTS_PER_POLICY, REQUEST_FAILURE,
    REQUEST_LATENCY, REQUEST_SUCCESS, ROUTING_POLICY_USAGE, TOKEN_USAGE,
};
use crate::stream::ReqwestStreamAdapter;
use crate::triton::{InferInputTensor, InferInputs, Output};
use bytes::Bytes;
use http::StatusCode;
use http_body_util::{combinators::BoxBody, BodyExt, Full};
use hyper::body::Incoming;
use hyper::{Method, Request, Response, Uri};
use log::{debug, error, info};
use prometheus::{gather, Encoder, TextEncoder};
use reqwest::header::{HeaderMap, HeaderValue, ACCEPT, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Mutex;

fn print_config(config: &RouterConfig) {
    debug!("{:#?}", config);
}

fn extract_forward_uri_path_and_query(req: &Request<Incoming>) -> Result<Uri, GatewayApiError> {
    let uri = req
        .uri()
        .path_and_query()
        .map(|x| x.as_str())
        .unwrap_or("")
        .to_string()
        .parse::<Uri>()?;

    Ok(uri)
}

#[derive(Serialize, Deserialize, Debug)]
struct Message {
    role: String,
    content: String,
}

type Messages = Vec<Message>;

fn extract_messages(value: &Value) -> Option<Messages> {
    value
        .get("messages")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
}

fn convert_messages_to_text_input(messages: &Messages) -> String {
    let text_input = serde_json::to_string(messages).unwrap_or_default();
    shorten_string(&text_input, 2000)
}

fn get_last_message_for_triton(messages: &Messages) -> String {
    messages
        .last()
        .map(|msg| msg.content.clone())
        .unwrap_or_default()
}

fn shorten_string(s: &str, max_length: usize) -> String {
    let len = s.len();
    if len <= max_length {
        s.to_string()
    } else {
        s[len - max_length..].to_string()
    }
}

async fn choose_model(
    policy: &Policy,
    client: &reqwest::Client,
    text_input: &str,
    _threshold: f64,
) -> Result<usize, GatewayApiError> {
    info!("Using policy: {}", &policy.name);
    info!("Triton input text: {:#?}", &text_input);
    let text_tensor = InferInputTensor {
        name: "INPUT".to_string(),
        datatype: "BYTES".to_string(),
        shape: vec![1, 1],
        data: vec![vec![text_input.to_string()]],
    };

    let data = InferInputs {
        inputs: vec![text_tensor],
    };

    let url = policy.url.clone();
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    let response = client.post(url).headers(headers).json(&data).send().await?;
    info!("Triton classification response: {:#?}", response);

    if !response.status().is_success() {
        let status = response.status();
        let error_body = response.bytes().await?;
        error!(
            "Triton error response: {}",
            String::from_utf8_lossy(&error_body)
        );

        return Err(GatewayApiError::TritonError(format!(
            "Triton service error ({}): {}",
            status,
            String::from_utf8_lossy(&error_body)
        )));
    }

    // Parse successful response
    let response: Output = response.json().await.map_err(|e| {
        error!("Failed to parse Triton response: {:?}", e);
        GatewayApiError::TritonError(format!("Invalid Triton response: {}", e))
    })?;

    info!("Triton Output: {:#?}", response);

    let output_tensor = response.outputs.first().ok_or_else(|| {
        GatewayApiError::TritonError("No outputs returned from the Triton response".to_string())
    })?;

    let model_index = output_tensor
        .data
        .iter()
        .enumerate()
        .max_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .ok_or_else(|| {
            error!("Invalid probability distribution from Triton");
            GatewayApiError::InvalidTritonOutput(
                "Could not determine model selection from probability distribution".to_string(),
            )
        })?;

    info!("model_index chosen by classifier: {:#?}", model_index);
    Ok(model_index)
}

fn modify_model(value: Value, model: &str) -> Result<Value, GatewayApiError> {
    let mut json = value.clone();
    json["model"] = Value::String(model.to_string());
    Ok(json)
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "lowercase")]
enum RoutingStrategy {
    Manual,
    Triton,
}

#[derive(Serialize, Deserialize, Debug)]
struct NimLlmRouterParams {
    policy: String,
    routing_strategy: Option<RoutingStrategy>,
    model: Option<String>,
    threshold: Option<f64>,
}

fn extract_nim_llm_router_params(value: &Value) -> Option<NimLlmRouterParams> {
    value
        .get("nim-llm-router")
        .and_then(|v| serde_json::from_value(v.clone()).ok())
}

fn remove_nim_llm_router_params(mut value: Value) -> Value {
    value
        .as_object_mut()
        .map(|map| map.remove("nim-llm-router"));
    value
}

// This might break response if the stream_options is not supported by the model,
// if you want to use this function, please make sure the model supports it.
// fn include_usage(mut value: Value) -> Value {
//     if let Some(obj) = value.as_object_mut() {
//         // Only add stream_options if not already present
//         if !obj.contains_key("stream_options") && obj.contains_key("stream") {
//             obj.insert(
//                 "stream_options".to_string(),
//                 serde_json::json!({ "include_usage": true }),
//             );
//             info!("Added stream_options to request");
//         }
//     }
//     value
// }

pub fn config(
    config: RouterConfig,
) -> Result<Response<BoxBody<Bytes, GatewayApiError>>, GatewayApiError> {
    let config = config.sanitized();
    let json_vec = serde_json::to_vec(&config).expect("Serialization to JSON should succeed.");
    let body_bytes = Bytes::from(json_vec);

    let full_body = Full::from(body_bytes)
        .map_err(|never| match never {})
        .boxed();

    let client_res = Response::builder().status(200).body(full_body)?;

    info!("/config: {client_res:#?}");
    Ok(client_res)
}

pub fn health() -> Result<Response<BoxBody<Bytes, GatewayApiError>>, GatewayApiError> {
    let body = serde_json::json!({ "status": "OK" });
    let json_vec = serde_json::to_vec(&body).expect("Serialization to JSON should succeed.");
    let body_bytes = Bytes::from(json_vec);

    let full_body = Full::from(body_bytes)
        .map_err(|never| match never {})
        .boxed();

    let client_res = Response::builder().status(200).body(full_body)?;

    info!("/health: {client_res:#?}");
    Ok(client_res)
}

pub fn metrics() -> Result<Response<BoxBody<Bytes, GatewayApiError>>, GatewayApiError> {
    let encoder = TextEncoder::new();
    let metric_families = gather();
    let mut buffer = Vec::new();

    if let Err(err) = encoder.encode(&metric_families, &mut buffer) {
        error!("Metric encoding error: {:?}", err);
        let body = String::from("Failed to encode metrics");
        let json_vec = serde_json::to_vec(&body).expect("Serialization to JSON should succeed.");
        let body_bytes = Bytes::from(json_vec);

        let full_body = Full::from(body_bytes)
            .map_err(|never| match never {})
            .boxed();
        let client_res = Response::builder().status(500).body(full_body)?;
        return Ok(client_res);
    }

    let body_bytes = Bytes::from(buffer);
    let full_body = Full::from(body_bytes)
        .map_err(|never| match never {})
        .boxed();

    let client_res = Response::builder()
        .header("Content-Type", encoder.format_type())
        .status(200)
        .body(full_body)?;

    info!("/metrics: {client_res:#?}");
    Ok(client_res)
}

pub fn unavailable() -> Result<Response<BoxBody<Bytes, GatewayApiError>>, GatewayApiError> {
    let body = serde_json::json!({ "path": "Unavailable" });
    let json_vec = serde_json::to_vec(&body).expect("Serialization to JSON should succeed.");
    let body_bytes = Bytes::from(json_vec);

    let full_body = Full::from(body_bytes)
        .map_err(|never| match never {})
        .boxed();

    let client_res = Response::builder().status(404).body(full_body)?;

    info!("/: {client_res:#?}");
    Ok(client_res)
}

pub async fn handler(
    req: Request<Incoming>,
    cfg: RouterConfig,
) -> Result<Response<BoxBody<Bytes, GatewayApiError>>, GatewayApiError> {
    let uri_path = req.uri().path();
    info!("Received request for URI: {}", uri_path);

    match uri_path {
        "/config" => {
            info!("Routing to config handler");
            config(cfg)
        }
        "/health" => {
            info!("Routing to health handler");
            health()
        }
        "/metrics" => {
            info!("Routing to metrics handler");
            metrics()
        }
        "/v1/chat/completions" | "/completions" => {
            info!("Routing to proxy handler");
            proxy(req, cfg).await
        }
        _ => {
            info!("Routing to Unavailable Path");
            unavailable()
        }
    }
}

pub async fn proxy(
    req: Request<Incoming>,
    config: RouterConfig,
) -> Result<Response<BoxBody<Bytes, GatewayApiError>>, GatewayApiError> {
    let overall_start = Instant::now();
    let mut model_selection_time = 0.0;
    let llm_resp_time_holder = Arc::new(Mutex::new(0.0));

    NUM_REQUESTS.inc();

    let result = (async {
        print_config(&config);

        let forward_uri_path_and_query = extract_forward_uri_path_and_query(&req)?;
        info!("forward_uri_path_and_query: {forward_uri_path_and_query:#?}");

        let (parts, body) = req.into_parts();
        info!("parts: {parts:#?}");

        let body_bytes = body.collect().await?.to_bytes();
        info!("body_bytes: {body_bytes:#?}");

        let body_str = String::from_utf8_lossy(&body_bytes);
        info!("body_str: {:#?}", &body_str);
        let json: Value = serde_json::from_str(&body_str).unwrap_or(Value::Null);
        info!("json: {:#?}", &json);

        let is_stream = if parts.method == Method::POST
            && parts
                .headers
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                == Some("application/json")
        {
            json["stream"].as_bool().unwrap_or(false)
        } else {
            false
        };
        info!("is_stream: {is_stream:#?}");

        let messages = extract_messages(&json).unwrap_or_default();
        info!("messages: {:#?}", &messages);
        let text_input = convert_messages_to_text_input(&messages);
        info!("text_input: {:#?}", &text_input);

        let client = reqwest::Client::new();

        let policy_name = if let Some(nim_llm_router_params) = extract_nim_llm_router_params(&json)
        {
            nim_llm_router_params.policy
        } else {
            info!("No nim-llm-router params => Using default policy at index 0");
            config
                .get_policy_by_index(0)
                .ok_or_else(|| {
                    GatewayApiError::ModelNotFound("No policy found at index 0".to_string())
                })?
                .name
        };

        let policy = config
            .get_policy_by_name(policy_name.as_str())
            .ok_or_else(|| {
                GatewayApiError::ModelNotFound(format!("Policy not found: {}", policy_name))
            })?;

        REQUESTS_PER_POLICY
            .with_label_values(&[policy.name.as_str()])
            .inc();

        let routing_strategy =
            extract_nim_llm_router_params(&json).and_then(|params| params.routing_strategy);

        let model_index = match routing_strategy {
            Some(RoutingStrategy::Manual) => {
                ROUTING_POLICY_USAGE.with_label_values(&["manual"]).inc();
                if let Some(nim_llm_router_params) = extract_nim_llm_router_params(&json) {
                    let model = nim_llm_router_params.model.ok_or_else(|| {
                        GatewayApiError::ModelNotFound(
                            "No model specified for manual routing".to_string(),
                        )
                    })?;
                    match policy.llms.iter().position(|llm| llm.name == model) {
                        Some(index) => index,
                        None => {
                            let error_body = format!("Model not found: {}", model);
                            let body = Full::from(error_body.into_bytes())
                                .map_err(|never| match never {})
                                .boxed();

                            let error_response = Response::builder()
                                .status(StatusCode::NOT_FOUND)
                                .header(CONTENT_TYPE, "application/json")
                                .body(body)?;

                            return Ok(error_response);
                        }
                    }
                } else {
                    let error_body = "Manual routing strategy requires nim-llm-router params";
                    let body = Full::from(error_body.to_string().into_bytes())
                        .map_err(|never| match never {})
                        .boxed();

                    let error_response = Response::builder()
                        .status(StatusCode::BAD_REQUEST)
                        .header(CONTENT_TYPE, "application/json")
                        .body(body)?;

                    return Ok(error_response);
                }
            }
            Some(RoutingStrategy::Triton) => {
                ROUTING_POLICY_USAGE.with_label_values(&["triton"]).inc();
                let selection_start = Instant::now();
                let threshold = extract_nim_llm_router_params(&json)
                    .and_then(|params| params.threshold)
                    .unwrap_or(0.5);
                let triton_text = get_last_message_for_triton(&messages);
                match choose_model(&policy, &client, &triton_text, threshold).await {
                    Ok(index) => {
                        model_selection_time = selection_start.elapsed().as_secs_f64();
                        MODEL_SELECTION_TIME.observe(model_selection_time);
                        index
                    }
                    Err(e) => {
                        // Extract error details
                        let error_body = match e {
                            GatewayApiError::TritonError(msg)
                            | GatewayApiError::InvalidTritonOutput(msg) => msg,
                            _ => return Err(e),
                        };

                        // Get original status code from error message if available
                        let status = if let Some(status_start) = error_body.find('(') {
                            if let Some(status_end) = error_body.find(')') {
                                if let Ok(code) =
                                    error_body[status_start + 1..status_end].parse::<u16>()
                                {
                                    StatusCode::from_u16(code)
                                        .unwrap_or(StatusCode::SERVICE_UNAVAILABLE)
                                } else {
                                    StatusCode::SERVICE_UNAVAILABLE
                                }
                            } else {
                                StatusCode::SERVICE_UNAVAILABLE
                            }
                        } else {
                            StatusCode::SERVICE_UNAVAILABLE
                        };

                        // Create error response
                        let body = Full::from(error_body.into_bytes())
                            .map_err(|never| match never {})
                            .boxed();

                        let error_response = Response::builder()
                            .status(status)
                            .header(CONTENT_TYPE, "application/json")
                            .body(body)?;

                        return Ok(error_response);
                    }
                }
            }
            None => {
                return Err(GatewayApiError::NoRoutingStrategy);
            }
        };

        let chosen_llm = policy.get_llm_by_index(model_index).ok_or_else(|| {
            GatewayApiError::ModelNotFound(format!("LLM not found at index {}", model_index))
        })?;

        let chosen_classifier = policy.get_llm_name_by_index(model_index).ok_or_else(|| {
            GatewayApiError::ModelNotFound(format!("LLM not found at index {}", model_index))
        })?;

        info!("Chosen Classifier: {:#?}", &chosen_classifier);

        REQUESTS_PER_MODEL
            .with_label_values(&[chosen_llm.name.as_str()])
            .inc();

        let api_base = &chosen_llm.api_base;
        let api_key = &chosen_llm.api_key;
        let model = &chosen_llm.model;

        info!("api_base: {:#?}", api_base);
        info!("model: {:#?}", model);

        let json = remove_nim_llm_router_params(json);
        info!("json after removing nim llm router params: {json:?}");

        let json = modify_model(json, model)?;
        info!("json after modifying model: {:#?}", &json);

        // Turn on this line if you want to include usage options in the request
        // let json = if is_stream { include_usage(json) } else { json };
        // info!("json after including usage options: {:#?}", &json);

        let method = http::Method::POST;
        let mut headers = http::HeaderMap::new();
        headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", api_key))?,
        );

        let uri = format!("{}{}", api_base, forward_uri_path_and_query);
        let mut reqwest_request = client.request(method, uri).json(&json);
        info!("reqwest_request: {reqwest_request:#?}");

        for (name, value) in headers.iter() {
            reqwest_request = reqwest_request.header(name, value);
        }

        let llm_req_start = Instant::now();
        let reqwest_response = reqwest_request.send().await?;
        let current_llm_resp = llm_req_start.elapsed().as_secs_f64();
        {
            let mut guard = llm_resp_time_holder.lock().await;
            *guard = current_llm_resp;
        }
        LLM_RESPONSE_TIME
            .with_label_values(&[chosen_llm.name.as_str()])
            .observe(current_llm_resp);

        let status = reqwest_response.status();
        let headers = reqwest_response.headers().clone();

        // If status is not successful, pass through the error response
        if !status.is_success() {
            let error_body = reqwest_response.bytes().await?;

            let body = Full::from(error_body)
                .map_err(|never| match never {})
                .boxed();

            let mut error_response = Response::builder().status(status).body(body)?;
            *error_response.headers_mut() = headers;
            error_response.headers_mut().insert(
                "X-Chosen-Classifier",
                HeaderValue::from_str(&chosen_classifier).unwrap(),
            );
            error!("error_response: {error_response:#?}");
            return Ok(error_response);
        }

        if let Some(token_usage_val) = reqwest_response.headers().get("X-Token-Usage") {
            if let Ok(token_usage_str) = token_usage_val.to_str() {
                if let Ok(token_count) = token_usage_str.parse::<u64>() {
                    TOKEN_USAGE
                        .with_label_values(&[chosen_llm.name.as_str()])
                        .inc_by(token_count);
                }
            }
        }

        if is_stream {
            let stream = reqwest_response.bytes_stream();
            let body = ReqwestStreamAdapter {
                inner: Box::pin(stream),
                llm_name: chosen_llm.name.clone(),
            };
            let boxed_body = BoxBody::new(body);

            let mut client_res = Response::new(boxed_body);
            *client_res.status_mut() = status;
            *client_res.headers_mut() = headers;
            client_res.headers_mut().insert(
                "X-Chosen-Classifier",
                HeaderValue::from_str(&chosen_classifier).unwrap(),
            );
            Ok(client_res)
        } else {
            let body_bytes = reqwest_response.bytes().await?;
            let body_clone = body_bytes.clone();
            // Parse and track token usage for non-streaming response
            if let Ok(json) = serde_json::from_slice::<Value>(&body_clone) {
                track_token_usage(&json, &chosen_llm.name);
            }
            let body = Full::from(body_bytes)
                .map_err(|never| match never {}) // never happens
                .boxed();

            let mut client_res = Response::builder().status(status).body(body)?;
            *client_res.headers_mut() = headers;
            client_res.headers_mut().insert(
                "X-Chosen-Classifier",
                HeaderValue::from_str(&chosen_classifier).unwrap(),
            );
            info!("client_res: {client_res:#?}");
            Ok(client_res)
        }
    })
    .await;

    let overall_latency = overall_start.elapsed().as_secs_f64();
    REQUEST_LATENCY.observe(overall_latency);

    let llm_resp_time = *llm_resp_time_holder.lock().await;
    let proxy_overhead = overall_latency - llm_resp_time - model_selection_time;
    PROXY_OVERHEAD_LATENCY.observe(proxy_overhead);

    match &result {
        Ok(response) => {
            if response.status().is_success() {
                REQUEST_SUCCESS.inc();
            } else {
                let status_code = response.status().as_u16();
                let error_type = if (400..500).contains(&status_code) {
                    "4xx"
                } else if (500..600).contains(&status_code) {
                    "5xx"
                } else {
                    "other"
                };
                REQUEST_FAILURE.with_label_values(&[error_type]).inc();
            }
        }
        Err(_err) => {
            // Handle system-level errors (non-HTTP errors)
            REQUEST_FAILURE.with_label_values(&["system"]).inc();
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_shorten_string() {
        let s = "Hello, world!".to_string();
        let max_length = 5;
        let expected = "orld!".to_string();
        assert_eq!(shorten_string(&s, max_length), expected);
    }

    #[tokio::test]
    async fn test_shorten_string_empty() {
        let s = "".to_string();
        let max_length = 5;
        let expected = "".to_string();
        assert_eq!(shorten_string(&s, max_length), expected);
    }

    #[tokio::test]
    async fn test_shorten_string_longer() {
        let s = "Hello, world!".to_string();
        let max_length = 15;
        let expected = "Hello, world!".to_string();
        assert_eq!(shorten_string(&s, max_length), expected);
    }

    #[tokio::test]
    async fn test_shorten_string_equal() {
        let s = "Hello, world!".to_string();
        let max_length = 13;
        let expected = "Hello, world!".to_string();
        assert_eq!(shorten_string(&s, max_length), expected);
    }
}
// #[cfg(test)]
// mod tests2 {
//     use super::*;
//     use serde_json::json;

//     #[test]
//     fn test_include_usage_adds_when_missing() {
//         let input = json!({"stream": true});
//         let output = include_usage(input);
//         assert_eq!(output["stream_options"]["include_usage"], true);
//     }

//     #[test]
//     fn test_include_usage_preserves_existing() {
//         let input = json!({
//             "stream": true,
//             "stream_options": {"existing": "config"}
//         });
//         let output = include_usage(input);
//         assert!(output["stream_options"]["include_usage"].is_null());
//     }
// }
