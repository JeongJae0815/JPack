import A
import tensorflow as tf

flags=tf.app.flags
FLAGS=flags.FLAGS
flags.DEFINE_string('parkjeongjae','dehat','it is a test')
def asgdf():
    a=1*1

def main(unused_argv):
    print "main B"

if __name__ == '__main__' :
    print "B exceuted"
    tf.app.run() # this instructions runs main function
else:
    print "B imported"