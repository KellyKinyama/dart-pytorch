LD_LIBRARY_PATH=/usr/lib/wsl/lib dart run bin/gpt2/medium/run_gpu_api.dart --serve --port 8080

# then:
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/info
curl -s -X POST http://127.0.0.1:8080/generate \
  -H 'content-type: application/json' \
  -d '{"tokens":[464,995,318],"maxNewTokens":20,"temperature":0.8,"topK":40,"seed":42}'