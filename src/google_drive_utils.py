import os
import json
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from googleapiclient.errors import HttpError
import pickle

# Load environment variables from .env file
load_dotenv()

# Global service object
_drive_service = None

def authenticate_user():
    global _drive_service
    
    # If we already have a service object, return it
    if _drive_service:
        return _drive_service
        
    # Path to store credentials
    token_path = 'token.pickle'
    
    credentials = None
    
    # Check if we have stored credentials
    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            credentials = pickle.load(token)
    
    # If credentials are invalid or don't exist, create new ones
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            # Get the Google credentials from environment variables
            google_credentials = os.environ.get('GOOGLE_CREDENTIALS')
            
            if not google_credentials:
                raise ValueError("GOOGLE_CREDENTIALS environment variable is not set.")
            
            # Parse the credentials JSON string
            credentials_dict = json.loads(google_credentials)
            
            # Define a specific port for the local server
            PORT = 61917
            
            # Set up the flow with the specific redirect URI
            flow = InstalledAppFlow.from_client_config(
                credentials_dict,
                scopes=['https://www.googleapis.com/auth/drive.file'],
                redirect_uri=f'http://localhost:{PORT}/'
            )
            
            # Run the flow with a local server
            credentials = flow.run_local_server(port=PORT)
            
            # Save the credentials for future runs
            with open(token_path, 'wb') as token:
                pickle.dump(credentials, token)
    
    # Build and return the service
    _drive_service = build('drive', 'v3', credentials=credentials)
    return _drive_service

def authenticate_with_service_account():
    # Get the service account credentials from environment variable
    service_account_key = os.environ.get('GOOGLE_SERVICE_ACCOUNT_KEY')
    
    if not service_account_key:
        raise ValueError("GOOGLE_SERVICE_ACCOUNT_KEY environment variable is not set.")
    
    # Parse the credentials JSON string
    credentials_dict = json.loads(service_account_key)
    
    # Create credentials from the service account info
    credentials = service_account.Credentials.from_service_account_info(
        credentials_dict,
        scopes=['https://www.googleapis.com/auth/drive.file']
    )
    
    # Build and return the service
    return build('drive', 'v3', credentials=credentials)

# Upload a file or list of files to Google Drive
def upload_to_drive(file_paths, folder_id=None):
    # Get the authenticated service
    service = authenticate_with_service_account()
    
    # Convert to list if a single file path is provided
    if not isinstance(file_paths, list):
        file_paths = [file_paths]
    
    # Skip folder verification if you know it's creating issues
    # Instead, directly list files in the folder
    
    # Get existing files in the folder
    existing_files = {}
    try:
        query = f"'{folder_id}' in parents and trashed = false"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get('files', [])
        
        # Create a mapping of filename to file id
        for file in files:
            existing_files[file['name']] = file['id']
            
        print(f"Found {len(existing_files)} existing files in the folder")
    except HttpError as error:
        print(f"Error listing files in folder: {error}")
        # If we can't list files, create new ones
        existing_files = {}
    
    results = []
    for file_path in file_paths:
        try:
            file_name = os.path.basename(file_path)
            media = MediaFileUpload(file_path, mimetype='text/csv')
            
            # Check if file already exists
            if file_name in existing_files:
                # Update the existing file
                file_id = existing_files[file_name]
                file = service.files().update(
                    fileId=file_id,
                    media_body=media,
                    fields='id'
                ).execute()
                print(f"File '{file_name}' updated successfully. File ID: {file.get('id')}")
            else:
                # Create a new file
                file_metadata = {'name': file_name}
                if folder_id:
                    file_metadata['parents'] = [folder_id]
                
                file = service.files().create(
                    body=file_metadata, 
                    media_body=media, 
                    fields='id'
                ).execute()
                print(f"File '{file_name}' created successfully. File ID: {file.get('id')}")
            
            results.append(file.get('id'))
        except HttpError as error:
            print(f"Error uploading {file_path}: {error}")
    
    return results

# Example usage
if __name__ == '__main__':
    file_path = 'path/to/your/file.csv' 
    folder_id = os.getenv('GOOGLE_FOLDER_ID') 
    upload_to_drive(file_path, folder_id)