
uv:
	@echo "$(GREEN)Setting up UV environment...$(NC)"
	@bash setup.sh

download:
	wget --content-disposition https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/3.58.0/files/ngccli_linux.zip -O ngccli_linux.zip && unzip -o ngccli_linux.zip
	if [ -z "$$NGC_CLI_API_KEY" ]; then \
  		echo "NGC_CLI_API_KEY environment variable is not set. Please set this environment variable and run again."; \
		exit 1; \
	else \
  		echo "NGC_CLI_API_KEY environment variable is set."; \
	fi 
	ngc-cli/ngc registry model download-version "nvidian/task-llm-router:v1"
	ngc-cli/ngc registry model download-version "nvidian/complexity-llm-router:v1"
	cp -r task-llm-router_vv1/* routers/
	cp -r complexity-llm-router_vv1/* routers/
	echo "Models downloaded and stored in routers directory successfully."

up:
	docker compose up router-server router-controller --build -d

down:
	docker compose -f docker-compose.yaml down -v

app:
	docker compose up app --build -d
	echo "Visit the app at localhost:8008"

metrics: 
	docker compose up grafana --build -d
	echo "Prometheus is running on 9090. Grafana is 3000 with user: admin and password: secret"

loadtest:
	docker compose up locust --build -d

build-router:
	docker compose up router-builder --build -d
