import os.path
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from paths import WORKING_DIR, MISC_DIR
import os
import datetime as dt
import yaml

SCOPES = ["https://www.googleapis.com/auth/drive"]

class GoogleDrive:
    def __init__(self, folder_name):
        token_path = f"{WORKING_DIR}/token.json"
        creds = None
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                f"{WORKING_DIR}/credentials.json", SCOPES
                )   
                creds = flow.run_local_server(port=0)
            with open(token_path, "w") as token:
                token.write(creds.to_json())

        self.service = build('drive', 'v3', credentials=creds)
        self.folder_name = folder_name
        self.folder_id = None
        print('drive initialized..')
        
    def create_folder(self, folder_name):
        folder_metadata = {
        'name': f'{folder_name}',
        "mimeType": "application/vnd.google-apps.folder",
        }

        created_folder = self.service.files().create(
            body=folder_metadata,
            fields='id'
        ).execute()

        f_id = created_folder['id']
        print(f'Created folder with id: {f_id}')
        return f_id

    def get_folder_id(self):
        results = self.service.files().list(
            q=f"mimeType='application/vnd.google-apps.folder' and name='{self.folder_name}' and trashed=false",
            fields='nextPageToken, files(id, name)'
            ).execute()
        
        files = results.get('files', [])
        for file in files:
            return file['id']
        
        return None
    
    def upload_file(self, file_path):
        file_data = {
            'name': f'{os.path.basename(file_path)}',
            'parents': [self.folder_id]
        }

        media = MediaFileUpload(file_path)
        self.service.files().create(body=file_data,
                                    media_body=media,
                                    fields='id').execute()
        
        print('File uploaded.')
    
    def upload_folder(self):
        if self.get_folder_id() is not None:
            print("Folder already exists...")
        else:
            drive_folder = dt.datetime.now().strftime("%H:%M_%d-%m-%Y")
            self.folder_id = self.create_folder(drive_folder)
            files = os.listdir(self.folder_name)
            print('Uploading files...')
            for file in files:
                self.upload_file(f'{MISC_DIR}/{file}')

            print('Uploading complete...')