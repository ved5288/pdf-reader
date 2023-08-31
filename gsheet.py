from oauth2client.service_account import ServiceAccountCredentials
import gspread
import datetime
import json
import streamlit as st

keyfile_dict = json.loads(st.secrets["SHEET_SECRETS"])
SHEET_URL = st.secrets["SHEET_URL"]

def get_table_range(row_number: int, no_of_columns: int):
    last_column_number = chr(ord('A') + no_of_columns - 1)
    return 'A{}:{}{}'.format(row_number, last_column_number, row_number)

def push_messages_to_sheet(source, query, response):
    scope = ['https://www.googleapis.com/auth/spreadsheets']
    creds = ServiceAccountCredentials.from_json_keyfile_dict(keyfile_dict, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_url(SHEET_URL).sheet1
    data = sheet.get_all_values()
    header_row = ['TIMESTAMP', 'PDF_SOURCE', 'Query', 'Response']
    if len(data) == 0:
        sheet.append_row(header_row, table_range=get_table_range(1, len(header_row)))
        data = sheet.get_all_values()
    
    row_values = [datetime.datetime.now().isoformat(), source, query, response ]
    row_number = len(data) + 1
    sheet.append_row(row_values, table_range=get_table_range(row_number, len(header_row)))
    print("Successfully updated the google sheet. Source: " + source + " Query: " + query + " Response: " + response)