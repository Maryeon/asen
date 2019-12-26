DATAPATH = '../data'
DATASET = 'fashionAI'
NUM_TRIPLETS = 100000

LABEL_FILE = {
	'train': 'train/Annotations/label.csv',
	'valid': 'valid/Annotations/label.csv',
	'test': 'test/Annotations/label.csv'
}

IMG_DIR = {
	'train': 'train/',
	'valid': 'valid/',
	'test': 'test/'
}

CONDITIONS = [
	'skirt_length',
	'sleeve_length',
	'coat_length',
	'pant_length',
	'collar_design',
	'lapel_design',
	'neckline_design',
	'neck_design'
]

CATEGORY_NUM = {
	'skirt_length':6,
	'sleeve_length':9,
	'coat_length':8,
	'pant_length':6,
	'collar_design':5,
	'lapel_design':5,
	'neckline_design':10,
	'neck_design':5,
}

CATEGORY_TARGET = {
	"skirt_length": [
  	"Invisible",
	"Short Length",
	"Knee Length",
	"Midi Length",
	"Ankle Length",
	"Floor Length"],
  "coat_length": [
  	"Invisible",
	"High Waist Length",
	"Regular Length",
	"Long Length",
	"Micro Length",
	"Knee Length",
	"Midi Length",
	"Ankle&Floor Length"],
  "collar_design": [
  	"Invisible",
	"Shirt Collar",
	"Peter Pan",
	"Puritan Collar",
	"Rib Collar"],
  "lapel_design": [
  	"Invisible",
	"Notched",
	"Collarless",
	"Shawl Collar",
	"Plus Size Shawl"],
  "neck_design": [
  	"Invisible",
	"Turtle Neck",
	"Ruffle Semi-High Collar",
	"Low Turtle Neck",
	"Draped Collar"],
  "neckline_design": [
  	"Invisible",
	"Strapless Neck",
	"Deep V Neckline",
	"Straight Neck",
	"V Neck",
	"Square Neckline",
	"Off Shoulder",
	"Round Neckline",
	"Sweat Heart Neck",
	"OneShoulder Neckline"],
  "pant_length": [
  	"Invisible",
	"Short Pant",
	"Mid Length",
	"3/4 Length",
	"Cropped Pant",
	"Full Length"],
  "sleeve_length": [
  	"Invisible",
	"Sleeveless",
	"Cup Sleeves",
	"Short Sleeves",
	"Elbow Sleeves",
	"3/4 Sleeves",
	"Wrist Length",
	"Long Sleeves",
	"Extra Long Sleeves"]
}
