FROM python:3.8-slim

WORKDIR /usr/src/app

# Copy the wheel file into the container
COPY dist/whisper_subtitle-0.1.0-py3-none-any.whl ./

# Install the package
RUN pip install whisper_subtitle-0.1.0-py3-none-any.whl

ENTRYPOINT ["whisper-subtitles"]
CMD ["--help"]
