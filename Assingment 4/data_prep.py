import os, csv, re

# some regex magic
def clean_text(text):

    text = re.sub(r'[^A-z0-9\n\t ]', '', text).strip('\n')
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

# cleans and combines training and validation data
def combine_csv(input, output):
    csv_files = [file for file in os.listdir(input)]

    with open(output, 'w', newline='', encoding='utf-8') as ofile:
        writer = csv.writer(ofile)

        writer.writerow(['Lyrics', 'Song Title', 'Genre'])

        for csv_file in csv_files:
            csv_file_path = os.path.join(input, csv_file)
            with open(csv_file_path,'r', newline='', encoding='utf-8') as infile:
                reader = csv.reader(infile)

                next(reader)

                for row in reader:
                    clean_row = [clean_text(cell) for cell in row]
                    writer.writerow(clean_row)

# cleans and prepares test data
def combine_test_data(input, output):
    data = []
    files = os.listdir(input)
    for file in files:
        songs = os.listdir(input+'/'+file)
        for song in songs:
            with open(input+'/'+file+'/'+song, 'r', newline='', encoding='utf-8') as s:
                lyrics = clean_text(s.read())
                song_title = re.sub(r'\.txt$', '', song)
                genre = file
                if lyrics is not None:
                    data.append((lyrics, song_title, genre))
    
    
    with open(output, 'w', newline='', encoding='utf-8') as ofile:
        writer = csv.writer(ofile)
        writer.writerow(['Lyrics', 'Song Title', 'Genre'])
        for lyrics, title, genre in data:
            writer.writerow([lyrics, title, genre])

# execution flow  
def main():
    combine_csv('testing', 'clean_lyrics.csv')
    combine_test_data('Test Songs', 'test_data.csv')

main()