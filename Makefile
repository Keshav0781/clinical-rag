# ─────────────────────────────────────────────
# ClinicalRAG — Makefile
# Shortcuts for common development tasks
# ─────────────────────────────────────────────

# Variables
IMAGE_NAME = clinical-rag
IMAGE_TAG = v1
PROJECT_ID = project-df0cbdfe-9de3-4681-b3b
REGION = europe-west1
REPO_NAME = clinical-rag-repo
SERVICE_NAME = clinical-rag-service

# ─────────────────────────────────────────────
# Local Development
# ─────────────────────────────────────────────

build:
	docker build --platform linux/amd64 -t $(IMAGE_NAME):$(IMAGE_TAG) .

run-local:
	docker run --platform linux/amd64 \
		-p 8080:8080 \
		-e GROQ_API_KEY=$$(grep GROQ_API_KEY .env | cut -d '=' -f2) \
		$(IMAGE_NAME):$(IMAGE_TAG)

# ─────────────────────────────────────────────
# GCP Deployment
# ─────────────────────────────────────────────

push:
	docker tag $(IMAGE_NAME):$(IMAGE_TAG) \
		$(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(REPO_NAME)/$(IMAGE_NAME):$(IMAGE_TAG)
	docker push \
		$(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(REPO_NAME)/$(IMAGE_NAME):$(IMAGE_TAG)

deploy:
	gcloud run deploy $(SERVICE_NAME) \
		--image $(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(REPO_NAME)/$(IMAGE_NAME):$(IMAGE_TAG) \
		--platform managed \
		--region $(REGION) \
		--allow-unauthenticated \
		--set-env-vars GROQ_API_KEY=$$(grep GROQ_API_KEY .env | cut -d '=' -f2)

# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

logs:
	gcloud run services logs read $(SERVICE_NAME) --region $(REGION)

health:
	curl https://$(SERVICE_NAME)-$(PROJECT_ID).$(REGION).run.app/health

.PHONY: build run-local push deploy logs health
