from bs4 import BeautifulSoup
import re
import requests
import os

saved = 0

def extract_genius_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    h1 = soup.find('h1')
    
    title = '' if not h1 else h1.text  
    
    # Find the div with data-lyrics-container="true"
    containers = soup.find_all('div', {'data-lyrics-container': 'true'})
    if not containers:
        return '', ''
    
    lyrics = ''
    for div in containers:
        for br in div.find_all('br'):
            br.replace_with('\n')
        # Get text content
        text = div.get_text()
        # Split text into lines
        lines = text.split('\n')
        # Remove lines that are just [some word]
        cleaned_lines = [line for line in lines if not re.match(r'^\[.*\]$', line.strip())]
        # Join the cleaned lines back into a single string
        result = '\n'.join(cleaned_lines)
        lyrics = lyrics + '\n' + result
    return title, lyrics.strip()


def find_tracks(tracklist):
    soup = BeautifulSoup(tracklist, 'html.parser')
    tracks = soup.find_all('search-result-item')
    
    result = []
    for track in tracks:
        subtitle_div = track.find('div', class_='mini_card-subtitle')
        if subtitle_div.text.strip() == 'Money Boy': # only Money Boy songs
            a_tag = track.find('a')
            if a_tag and 'href' in a_tag.attrs:
                link = a_tag['href']
                result.append(link)
        
    return result

def scrape_song(url, filename='songs.txt'):
    # Send a GET request to the URL
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return

    # Extract lyrics text from the HTML content
    html = response.text
    lyrics = extract_genius_text(html)

    if lyrics[1] == '':
        print("No lyrics found on page " + url)
        return

    # Check if the file exists and is not empty
    file_exists = os.path.isfile(filename)
    file_not_empty = file_exists and os.path.getsize(filename) > 0

    # Open the file in append mode
    with open(filename, 'a', encoding='utf-8') as f:
        if file_not_empty:
            # Add a delimiter line with dashes
            f.write('\n' + '-' * 40 + '\n')
        f.write(f'TITLE: {lyrics[0]}\n')
        f.write(lyrics[1])
        print(f'Saved lyrics for {lyrics[0]}')
        global saved 
        saved += 1


with open('genius_tracklist.html', 'r', encoding='utf-8') as f:
  tracklist = f.read()

track_links = find_tracks(tracklist)

for link in track_links:
    scrape_song(link)

print(f'Successfully scraped {saved} songs')