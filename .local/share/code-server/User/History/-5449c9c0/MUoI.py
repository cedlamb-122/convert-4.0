from loaders import load_csv_data, load_from_json
import os

if __name__ == '__main__':
    if os.environ.get('env', 'DEBUG') =='DEBUG':
        load_csv_data('data.csv')
    else:
        print('goodbye world')
