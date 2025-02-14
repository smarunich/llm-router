# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from locust import HttpUser, task, between
from locust.event import EventHook
from prometheus_client import start_http_server, Gauge, Counter
from pydantic import BaseModel
import random

# Prometheus metrics
request_counter = Counter('http_requests_total', 'Total requests made', ['endpoint', 'model'])
latency_gauge = Gauge('request_latency_seconds', 'Request latency', ['endpoint'])
tokens_counter = Counter('tokens_total', 'Total tokens used', ['model', 'type'])

class UsageTokens(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

# Event handling for metrics
event_hook = EventHook()

def process_request(usage_tokens: UsageTokens, model: str, **kwargs):
    tokens_counter.labels(model=model, type="prompt").inc(usage_tokens.prompt_tokens)
    tokens_counter.labels(model=model, type="completion").inc(usage_tokens.completion_tokens)
    tokens_counter.labels(model=model, type="total").inc(usage_tokens.total_tokens)

event_hook.add_listener(process_request)

# Test messages
MESSAGES = [
    "Write a song about space exploration",
    "Explain quantum computing to a 5-year old",
    "Create a short story about a magical library"
]

class APIUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def chat_completion(self):
        prompt = random.choice(MESSAGES)
        message = {"role": "user", "content": prompt}
        
        data = {
            "model": "",
            "messages": [message],
            "nim-llm-router": {
                "routing_strategy": "triton",
                "policy": "task_router",
                "threshold": 0.2,
            }
        }

        with self.client.post("/v1/chat/completions", json=data, catch_response=True) as response:
            try:
                if response.status_code == 200:
                    chat_completion = response.json()
                    usage = chat_completion.get("usage")
                    model = chat_completion.get("model")
                    
                    usage_tokens = UsageTokens(
                        prompt_tokens=usage.get("prompt_tokens"),
                        completion_tokens=usage.get("completion_tokens"),
                        total_tokens=usage.get("total_tokens")
                    )
                    
                    event_hook.fire(usage_tokens=usage_tokens, model=model)
                    request_counter.labels(endpoint="/v1/chat/completions", model=model).inc()
                    latency_gauge.labels(endpoint="/v1/chat/completions").set(response.elapsed.total_seconds())
                else:
                    error_msg = f"Request failed with status {response.status_code}"
                    try:
                        error_msg += f": {response.json()}"
                    except:
                        error_msg += f": {response.text}"
                    response.failure(error_msg)
            except Exception as e:
                response.failure(f"Exception during request: {str(e)}")
# Start Prometheus metrics server
start_http_server(4000, addr='0.0.0.0')
