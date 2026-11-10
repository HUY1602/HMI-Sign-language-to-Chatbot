# HMI Sign Language to Chatbot

Dự án thử nghiệm nhận diện một số ký hiệu tay từ webcam, ghép các ký hiệu thành câu và gửi câu đó tới chatbot. Dự án cũng có giao diện web Flask để trò chuyện bằng văn bản.

## Tính năng

- Thu thập ảnh ký hiệu tay từ webcam.
- Trích xuất 21 điểm mốc bàn tay bằng MediaPipe.
- Huấn luyện bộ phân loại Random Forest.
- Nhận diện ký hiệu theo thời gian thực bằng webcam.
- Gửi câu đã nhận diện tới chatbot Flask.
- Giao diện web cho phép gửi tin nhắn và nhận phản hồi từ Gemini.

> Đây là dự án thử nghiệm với bộ ký hiệu và nhãn được định nghĩa sẵn trong mã nguồn; chưa phải hệ thống dịch đầy đủ cho một ngôn ngữ ký hiệu.

## Cấu trúc dự án

```text
.
├── chatbot/
│   ├── app.py                  # Flask API và trang chatbot
│   ├── templates/
│   │   └── index.html
│   └── static/
│       ├── css/
│       └── js/
├── collect_imgs.py             # Thu thập ảnh từ webcam
├── create_dataset.py           # Trích xuất đặc trưng và tạo data.pickle
├── train_classifier.py         # Huấn luyện, tạo model.p
├── inference_classifier.py     # Nhận diện ký hiệu và gửi câu tới chatbot
└── requirements.txt
```

## Yêu cầu

- Python
- Webcam
- Kết nối Internet khi sử dụng chatbot Gemini
- Các thư viện Python trong `requirements.txt`

Một số thư viện cần cho chatbot và các script hiện chưa có trong `requirements.txt`. Có thể cài bổ sung bằng:

```bash
pip install Flask requests numpy matplotlib
```

Cài các thư viện còn lại:

```bash
pip install -r requirements.txt
```

## Huấn luyện mô hình nhận diện

Thực hiện các lệnh sau từ thư mục gốc của dự án.

### 1. Thu thập ảnh

```bash
python collect_imgs.py
```

Chương trình mở webcam và lưu ảnh vào `data/`, chia thành 8 lớp (`0` đến `7`), mỗi lớp dự kiến có 100 ảnh. Hãy chuẩn bị ký hiệu tương ứng cho từng lớp khi thu thập dữ liệu.

### 2. Tạo dữ liệu huấn luyện

```bash
python create_dataset.py
```

Lệnh này đọc ảnh trong `data/` và tạo file `data.pickle`.

### 3. Huấn luyện mô hình

```bash
python train_classifier.py
```

Sau khi huấn luyện, chương trình tạo file `model.p`.

### 4. Chạy nhận diện

```bash
python inference_classifier.py
```

Ứng dụng sử dụng webcam, hiển thị ký hiệu dự đoán và gửi câu đã nhận diện tới chatbot khi phát hiện ký hiệu `ok`. Nhấn `q` trên cửa sổ webcam để thoát.

> `inference_classifier.py` cần file `model.p` ở thư mục gốc. Hãy bảo đảm nhãn và thứ tự lớp trong dữ liệu huấn luyện khớp với `labels_dict` trong script.

## Chạy giao diện chatbot

Khởi chạy server Flask từ thư mục gốc:

```bash
python chatbot/app.py
```

Mở địa chỉ sau trong trình duyệt:

```text
http://127.0.0.1:5000
```

Giao diện gửi tin nhắn tới endpoint `POST /chat`; backend chuyển nội dung tới Gemini và trả phản hồi cho trình duyệt.

## Lưu ý

- Cần cấp quyền truy cập webcam cho ứng dụng.
- Dữ liệu ảnh, `data.pickle` và `model.p` cần được tạo trước khi chạy nhận diện; các tệp này không được tạo tự động khi khởi động chatbot.
- Các nhãn trong `inference_classifier.py` hiện được khai báo cố định. Nếu thay đổi dữ liệu hoặc ký hiệu, cần cập nhật nhãn tương ứng.
- Server Flask đang chạy ở chế độ debug trong mã nguồn; không dùng cấu hình này cho môi trường production.
- Chỉ sử dụng dữ liệu và hình ảnh mà bạn có quyền thu thập.
