# Aletheia web edition for Hugging Face Spaces (Docker). Serves the Aletheia screen with Voila on port 7860.
FROM python:3.12-slim
RUN useradd -m -u 1000 user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user . .
USER user
EXPOSE 7860
CMD ["voila", "app.ipynb"]
