import os
import google.auth
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

# Lấy credentials từ file client_secret.json (tệp bạn đã tải về từ Google Cloud Console)
creds = None
if os.path.exists('token.json'):
    creds = Credentials.from_authorized_user_file('token.json', ['https://www.googleapis.com/auth/cloud-platform'])

# Nếu không có (hoặc credentials hết hạn), yêu cầu người dùng đăng nhập lại
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        from google_auth_oauthlib.flow import InstalledAppFlow
        flow = InstalledAppFlow.from_client_secrets_file(
            'client_secret_594765464944-2dcoh5ni4ds4tp76bbsmgfiohf83ckoi.apps.googleusercontent.com.json',  # Đảm bảo bạn đã tải xuống tệp này từ Google Cloud Console
            ['https://www.googleapis.com/auth/cloud-platform']
        )
        creds = flow.run_local_server(port=0)

    # Lưu credentials để tái sử dụng sau này
    with open('token.json', 'w') as token:
        token.write(creds.to_json())

# Lấy Access Token từ credentials
access_token = creds.token
print(f"Access Token: {access_token}")
