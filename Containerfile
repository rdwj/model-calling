FROM registry.access.redhat.com/ubi9/python-312:latest

# Set proper labels
LABEL name="model-calling-service"
LABEL description="OpenAI-compatible API service for multiple LLM backends"
LABEL maintainer="Wes Jackson"

USER 0

# Work in app directory
WORKDIR /opt/app-root/src

# Copy the application and install the package (examples are not included)
COPY . .
RUN pip install --no-cache-dir .

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
