import zipfile
import io
from src.main import Parameters


class Compression:
    def __init__(self):
        pass

    @staticmethod
    def write_data_to_files(inp_data, file_name):
        print(f" *** Writing the data to - {file_name}")
        throwaway_storage = io.StringIO(inp_data)
        with open(Parameters.file_name, 'w') as f:
            for line in throwaway_storage:
                f.write(line)

    @staticmethod
    def file_compress(inp_file_names, out_zip_file):

        compression = zipfile.ZIP_DEFLATED

    print(f" *** Input File name passed for zipping - {Parameters.inp_file_names}")

    print(f' *** out_zip_file is - {Parameters.out_zip_file}')
    zf = zipfile.ZipFile(Parameters.out_zip_file, mode="w")

    try:
        for file_to_write in Parameters.inp_file_names:
         print(f' *** Processing file {file_to_write}')
        zf.write(file_to_write, file_to_write, compress_type=Parameters.compression)

    except FileNotFoundError as e:
     print(f' *** Exception occurred during zip process - {e}')
    finally:
      zf.close()
