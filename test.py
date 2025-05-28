from paths import MISC_DIR

with open(f'{MISC_DIR}/hyperparams.txt', 'r+') as file:
    file.seek(2)
    file.write("ya ali")