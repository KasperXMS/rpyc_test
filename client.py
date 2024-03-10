import rpyc

if __name__ == "__main__":
    c = rpyc.connect("192.168.1.132", 18861)
    print(c.root.get_answer())
    print(c.root.get_question())
    print(c.root.the_real_answer_though)
    c.close()