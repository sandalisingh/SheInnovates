from ImageAnalyzer import ImageAnalyzer
import Configurations as CF

analyzer = ImageAnalyzer(image_path=CF.IP_IMG_PATH_OBJ_DET)
analyzer.run()