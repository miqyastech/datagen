import gdown

URL_FEMALE = 'https://drive.google.com/file/d/1Qt4L_Yyqeu1x0ju1d65VWP_by-T_4N4f/view?usp=sharing'
OUTPUT_FEMALE = 'basicModel_f_lbs_10_207_0_v1.1.0.pkl'
gdown.download(URL_FEMALE, OUTPUT_FEMALE, quiet=False, fuzzy=True)

URL_MALE = 'https://drive.google.com/file/d/1NvC8SXPGqYy5X2qzW8fJi335SauHNUwq/view?usp=sharing'
OUTPUT_MALE = 'basicModel_m_lbs_10_207_0_v1.1.0.pkl'
gdown.download(URL_MALE, OUTPUT_MALE, quiet=False, fuzzy=True)