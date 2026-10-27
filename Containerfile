FROM registry.access.redhat.com/ubi9/python-312:9.8-1792441557@sha256:f43a57e9a6a2243e333008219002b9f2b1661868a94199f5dec4d602e1501482

# Set proper labels
LABEL name="model-calling-service"
LABEL description="OpenAI-compatible API service for multiple LLM backends"
LABEL maintainer="Wes Jackson"

USER 0

# Work in app directory
WORKDIR /opt/app-root/src

# uv comes from a digest-pinned image and installs exactly what uv.lock
# specifies. Base and uv images are pinned by digest; see "Dependency Policy"
# in CONTRIBUTING.md.
COPY --from=ghcr.io/astral-sh/uv:0.13.3@sha256:bee0a2a458fd956c5df7b8b4c5fed2257b0af34866e3da7b51db0be3acb375d1 /uv /usr/local/bin/uv
ENV UV_PROJECT_ENVIRONMENT=/opt/app-root \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1

# Copy the application and install it with its locked dependencies
# (examples and dev tools are not included)
COPY . .
RUN uv sync --frozen --no-dev --no-editable --inexact

# Change permissions to the non-root user
RUN chown -R 1001:0 /opt/app-root/src && \
    chmod -R g=u /opt/app-root/src

# Switch back to non-root user
USER 1001

# Expose the port
EXPOSE 8000

# Environment variables
ENV PORT=8000
ENV OLLAMA_BASE_URL=http://localhost:11434
ENV VLLM_BASE_URL=http://localhost:8000

# Run the server
CMD ["python", "-m", "model_calling"]
