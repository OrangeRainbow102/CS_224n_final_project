import pickle



def main():
    with open('pretrain_ce_small_synthetic5.pkl', 'rb') as file:
        results = pickle.load(file)
    print(len(results))




if __name__ == '__main__':
    main()