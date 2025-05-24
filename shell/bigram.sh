#!/bin/bash
cd ..

# bigram LDA 실행 스크립트
python main.py --category 뉴스 --num_topics 8 --iterations 100 --ngram 2 --seed 11
python main.py --category 뉴스 --num_topics 8 --iterations 100 --ngram 2 --seed 22
python main.py --category 뉴스 --num_topics 8 --iterations 100 --ngram 2 --seed 33
python main.py --category 뉴스 --num_topics 8 --iterations 100 --ngram 2 --seed 44
python main.py --category 뉴스 --num_topics 8 --iterations 100 --ngram 2 --seed 55

python main.py --category 위키 --num_topics 8 --iterations 100 --ngram 2 --seed 11
python main.py --category 위키 --num_topics 8 --iterations 100 --ngram 2 --seed 22
python main.py --category 위키 --num_topics 8 --iterations 100 --ngram 2 --seed 33
python main.py --category 위키 --num_topics 8 --iterations 100 --ngram 2 --seed 44
python main.py --category 위키 --num_topics 8 --iterations 100 --ngram 2 --seed 55