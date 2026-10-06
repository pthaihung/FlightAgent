# Benchmark OpenRouter API thật

Model lập kế hoạch, chọn chuyến và tạo hành động qua API thật; tool chuyến bay vẫn là mock. Harness kiểm schema, ràng buộc, quyền và hoàn thành bằng code.

## Cấu hình

Điền `.env` từ `.env.example`:

```env
OPENROUTER_API_KEY=key_cua_ban
OPENROUTER_MODEL=provider/model
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Dùng ID đầy đủ của model hỗ trợ tool calling/structured output. Không đưa key vào chat hoặc `.env` lên GitHub. Bộ đã nộp dùng `qwen/qwen3.7-flash`.

```powershell
python -m pip install -r requirements.txt
python main.py demo --model openrouter --pattern hybrid --runtime langgraph --output results_local/smoke.json
python main.py benchmark --model openrouter --runtime langgraph --repeats 1 --output results_local/api
```

Một seed có 36 lượt: 12 tình huống × 3 mẫu. Ngân sách mặc định 180 giây mỗi lượt; lưu kết quả sau từng lượt.

## Reasoning và schema

Cấu hình OpenRouter là `reasoning={"effort":"none"}`. SDK đang cài bỏ trường `enabled`, nên test kiểm serialize để xác nhận tham số gửi provider. Manifest và kết quả ghi `reasoning_requested`; đây là cấu hình yêu cầu, không đo suy luận nội bộ.

Schema riêng mỗi tool cấm trường thừa. `search_flights` nhận `args={}` vì harness giữ task bất biến; quote/book/lookup nhận ID tương ứng. Model vẫn có thể thiếu structured output hoặc chọn ID sai, được ghi là thất bại.

## Tiếp tục sau lỗi API

```powershell
python main.py benchmark --model openrouter --runtime langgraph --repeats 1 --output results_local/api --resume
```

`--resume` bỏ qua mọi lượt đã đánh giá, kể cả lượt Agent thất bại. Lỗi quota/xác thực/kết nối lưu riêng trong `api_errors.json`, không tính vào điểm Agent. Giữ nguyên model, runtime, repeats, ngân sách, reasoning; nếu sửa workflow/schema hoặc đổi model thì dùng thư mục mới.

`TooManyRequestsResponseError: Provider returned error` cho biết API/provider giới hạn yêu cầu; thông báo này chưa xác định được hết quota ngày hay hết tiền.

## Dữ liệu đã nộp

`results_api/manifest.json`: **36/36 lượt**, `benchmark_complete=true`, `pending_runs=0`; một lần lỗi 429 rồi tiếp tục thành công.

- `runs.json`: trace, verifier, usage từng lượt.
- `runs.csv`: chỉ số mỗi lượt.
- `summary.json`, `comparison.md`: tổng hợp ba mẫu.
- `api_errors.json`: lịch sử lỗi hạ tầng.

Chỉ sử dụng token khi `usage_available=true`. Token của lượt đánh giá không gồm lần API lỗi đã tách riêng, nên không đại diện toàn bộ hóa đơn. Chi phí cần đối chiếu billing và đơn giá tại ngày chạy.

## Tài liệu

- [Tích hợp LangChain với OpenRouter](https://openrouter.ai/blog/tutorials/langchain-chatopenrouter-setup/)
- [Reasoning OpenRouter](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens)
- [Danh mục model](https://openrouter.ai/models)
