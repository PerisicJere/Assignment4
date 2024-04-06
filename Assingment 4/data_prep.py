import os, csv, re

def clean_text(text):

    text = re.sub(r'[^A-z0-9\n\t ]', '', text)
    text = text.lower().strip()

    return text

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

def main():
    input = 'testing'
    output = 'clean_lyrics.csv'
    combine_csv(input, output)

main()