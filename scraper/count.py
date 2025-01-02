import os

def count_songs_and_lines(filename='songs.txt'):
    """
    Counts the number of songs and the total number of lyric lines in the specified file.

    Args:
        filename (str): The path to the file containing the scraped songs. Defaults to 'songs.txt'.
    
    Returns:
        None
    """
    if not os.path.isfile(filename):
        print(f"Error: The file '{filename}' does not exist.")
        return
    
    song_count = 0
    total_lines = 0
    current_song = False  # Flag to indicate if we are reading lyrics of a current song

    try:
        with open(filename, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, start=1):
                stripped_line = line.strip()
                
                # Check for the start of a new song
                if stripped_line.startswith('TITLE: '):
                    song_count += 1
                    current_song = True
                    continue  # Move to the next line

                # Check for delimiter indicating the end of a song
                if stripped_line.startswith('-' * 40):
                    current_song = False
                    continue  # Move to the next line

                # If we're within a song's lyrics, count the lines
                if current_song:
                    if stripped_line:  # Ignore empty lines
                        total_lines += 1

        print(f"Number of songs downloaded: {song_count}")
        print(f"Total number of lyric lines: {total_lines}")

    except Exception as e:
        print(f"An error occurred while processing the file: {e}")

if __name__ == "__main__":
    count_songs_and_lines()
