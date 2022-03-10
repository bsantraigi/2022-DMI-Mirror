import hashlib
import datetime

def gen_uniq_run_id():
    T = datetime.datetime.now().timestamp()
    print(T)
    source = str(T).encode()
    print(source)
    md5 = hashlib.md5(source).hexdigest().upper()  # returns a str
    return md5[:8]


if __name__=="__main__":
    x = gen_uniq_run_id()
    print(x)