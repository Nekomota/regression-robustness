import requests
from bs4 import BeautifulSoup
import csv

# Define the output CSV file path
output_path = r"C:\Users\nickc\OneDrive\Desktop\royalroad_data9.csv"

def scrape_fiction(fiction_id):
    url = f"https://www.royalroad.com/fiction/{fiction_id}"
    response = requests.get(url)

    if response.status_code == 404:
        return None  # Skip this ID

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract title
    title_element = soup.find("h1", class_="font-white")
    title = title_element.text.strip() if title_element else "N/A"

    # Extract synopsis
    synopsis_div = soup.find("div", class_="description")
    synopsis = synopsis_div.text.strip() if synopsis_div else "N/A"

    # Extract followers
    followers_label = soup.find("li", string="Followers :")
    followers = followers_label.find_next_sibling("li").text.strip() if followers_label else "N/A"

    return (fiction_id, title, synopsis, followers)

print("Starting data scraping...")

# Open CSV file and write header
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Fiction ID", "Title", "Synopsis", "Followers"])  # CSV header

    for fiction_id in range(23999, 29999):  # Limited range for testing
        data = scrape_fiction(fiction_id)
        if data:
            writer.writerow(data)

print("Scraping completed. Data saved to:", output_path)
