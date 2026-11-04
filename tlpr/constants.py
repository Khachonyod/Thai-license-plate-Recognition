from pathlib import Path

# Project root directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Default model weights
DEFAULT_PLATE_MODEL = PROJECT_DIR / "weights" / "plate_detector.pt"
DEFAULT_CHAR_MODEL = PROJECT_DIR / "weights" / "cChar_detector.pt"

# Digit class names
DIGIT_NAMES = {
    "Zero", "One", "Two", "Three", "Four",
    "Five", "Six", "Seven", "Eight", "Nine",
}

# Mapping YOLO class name to Thai character or digit
CLASS_NAME_TO_CHAR = {
    # Digits
    "Zero": "0", "One": "1", "Two": "2", "Three": "3", "Four": "4",
    "Five": "5", "Six": "6", "Seven": "7", "Eight": "8", "Nine": "9",
    # Thai Consonants (44 characters)
    "gor-gai": "ก", "kor-kai": "ข", "kor-kuad": "ฃ", "kor-kwai": "ค", "kor-khon": "ฅ",
    "kor-ra-kang": "ฆ", "ngor-ngu": "ง", "jor-jarn": "จ", "chor-ching": "ฉ", "chor-chang": "ช",
    "sor-so": "ซ", "chor-ga-cher": "ฌ", "yor-ying": "ญ", "dor-cha-da": "ฎ", "tor-pa-tak": "ฏ",
    "thor-than": "ฐ", "thor-mon-tho": "ฑ", "thor-pud-thao": "ฒ", "nor-nen": "ณ", "dor-dek": "ด",
    "tor-tao": "ต", "thor-thung": "ถ", "thor-ta-harn": "ท", "thor-thong": "ธ", "nor-nu": "น",
    "bor-bai-mai": "บ", "por-pla": "ป", "por-pung": "ผ", "for-fa": "ฝ", "por-parn": "พ",
    "for-fan": "ฟ", "por-sam-phao": "ภ", "mor-ma": "ม", "yor-yak": "ย", "ror-ruea": "ร",
    "lor-ling": "ล", "vor-van": "ว", "sor-sa-la": "ศ", "sor-rue-si": "ษ", "sor-suea": "ส",
    "hor-heeb": "ห", "ror-ju-la": "ฬ", "or-ang": "อ", "hor-nok-hooke": "ฮ",
}

# Visual confusion mappings between Thai consonants and digits
THAI_TO_DIGIT_CONFUSION = {
    "or-ang": "0", "thor-thung": "0", "bor-bai-mai": "0", "dor-dek": "0", "vor-van": "0",
    "ror-ruea": "1", "gor-gai": "1", "for-fan": "1", "for-fa": "1",
    "kor-kai": "2", "chor-chang": "2", "sor-so": "2",
    "sor-sa-la": "4", "sor-rue-si": "4",
    "thor-thong": "5",
    "dor-cha-da": "6",
    "thor-ta-harn": "7", "tor-tao": "7",
    "mor-ma": "8", "yor-yak": "8", "hor-heeb": "8", "por-parn": "8",
    "kor-kwai": "9", "ngor-ngu": "9",
}

DIGIT_TO_THAI_CONFUSION = {
    "Zero": "อ", "One": "ก", "Two": "ข", "Three": "ว", "Four": "ศ",
    "Five": "ธ", "Six": "ด", "Seven": "ท", "Eight": "ม", "Nine": "ค",
}

# 77 Official Provinces of Thailand
PROVINCES_77 = [
    "กรุงเทพมหานคร", "กระบี่", "กาญจนบุรี", "กาฬสินธุ์", "กำแพงเพชร", "ขอนแก่น", "จันทบุรี", "ฉะเชิงเทรา",
    "ชลบุรี", "ชัยนาท", "ชัยภูมิ", "ชุมพร", "เชียงราย", "เชียงใหม่", "ตรัง", "ตราด", "ตาก", "นครนายก",
    "นครปฐม", "นครพนม", "นครราชสีมา", "นครศรีธรรมราช", "นครสวรรค์", "นนทบุรี", "นราธิวาส", "น่าน",
    "บึงกาฬ", "บุรีรัมย์", "ปทุมธานี", "ประจวบคีรีขันธ์", "ปราจีนบุรี", "ปัตตานี", "พระนครศรีอยุธยา",
    "พะเยา", "พังงา", "พัทลุง", "พิจิตร", "พิษณุโลก", "เพชรบุรี", "เพชรบูรณ์", "แพร่", "ภูเก็ต",
    "มหาสารคาม", "มุกดาหาร", "แม่ฮ่องสอน", "ยโสธร", "ยะลา", "ร้อยเอ็ด", "ระนอง", "ระยอง", "ราชบุรี",
    "ลพบุรี", "ลำปาง", "ลำพูน", "เลย", "ศรีสะเกษ", "สกลนคร", "สงขลา", "สตูล", "สมุทรปราการ",
    "สมุทรสงคราม", "สมุทรสาคร", "สระแก้ว", "สระบุรี", "สิงห์บุรี", "สุโขทัย", "สุพรรณบุรี", "สุราษฎร์ธานี",
    "สุรินทร์", "หนองคาย", "หนองบัวลำภู", "อ่างทอง", "อำนาจเจริญ", "อุดรธานี", "อุตรดิตถ์", "อุทัยธานี",
    "อุบลราชธานี",
]

