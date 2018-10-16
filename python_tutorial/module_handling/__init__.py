# __init__.py는 package안에 있는 모듈들을 관리해주는 파일로,
# python3.4부터는 굳이 안써도 실행이 되지만, 전체적인 호환을 위해 이파일을 생성해주는것이 좋다.
#from .model import DeepLabResNetModel
#from .image_reader import ImageReader #image_reader 모듈안에 있는 ImageReader함수를 불러오는 
#from .utils import decode_labels, inv_preprocess, prepare_label


#__all__ = ['image_reader1','image_reader2'] 
# from moduleTeset import * 명령어를 쓰면 위에 적혀있는 모듈을 전부 import하게 된다.


#from .image_reader1 import a
#from .image_reader2 import b
#그러니깐 위의 코드는 원래 모듈안의 함수a,b를 불러오려면
# from moduleTest.image_reader1 import a 이렇게 해야하는데
# 위처럼 써줌으로써 함수 a,b를 불러올때 from moduleTest import a  와 같이 InputHandler를 생략할수 있다.

