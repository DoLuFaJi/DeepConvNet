from settings import CLASSIFIED_TEST_DATA_GOOGLE, CLASSIFIED_TEST_DATA_GOOGLE2, CLASSIFIED_TEST_DATA_YALE, TEST_DATA_GOOGLE, TEST_DATA_GOOGLE_2, TEST_DATA_YALE, CLASSIFIED_TEST_DATA

def create_posneg():
    yale_file = open(CLASSIFIED_TEST_DATA_YALE, 'w')
    google_file = open(CLASSIFIED_TEST_DATA_GOOGLE, 'w')
    google_file_2 =  open(CLASSIFIED_TEST_DATA_GOOGLE2, 'w')
    with open(CLASSIFIED_TEST_DATA, 'r') as posneg_file:
        for line in posneg_file:
            if TEST_DATA_GOOGLE_2 in line:
                google_file_2.write(line)
            if TEST_DATA_GOOGLE in line:
                google_file.write(line)
            if TEST_DATA_YALE in line:
                yale_file.write(line)
    yale_file.close()
    google_file.close()
    google_file_2.close()

create_posneg()
