---
layout: post
category: MODEL
---

## BERT Fine-Tuning이란?
   BERT는 대용량의 텍스트 코퍼스의 단어 임베딩을 MLM과 NSP 방식을 통해 사전학습한 transformer 모델입니다. MLM과 NSP task를 잘 수행하도록 학습된 BERT는 언어의 맥락적, 문법적 특징을 이해할 수 있다고 여겨집니다. 이와 같이 사전학습된 모델을 다시 해결하고자 하는 타겟 task를 잘 수행하도록 모델의 파라미터를 재조정하는 과정이 fine-tuning 과정입니다.<br/><br/> 이 글에서는 multi-class text classification을 잘하도록 BERT 파라미터를 fine-tuning하는 코드를 리뷰하겠습니다.
<br/>

## BERT Fine-tuning의 main함수 구성

### 1. Configuration 초기화
1) BERT 자체 configuration
```
bert_cfg = train.Config.from_json(train_cfg)
```
BERT를 사용하기 위해서는 기존에 사전학습에 사용된 하이퍼파라미터들이 필요합니다. 임베딩 차원(dim), 모델 구성 layer의 수(n_layers), input 데이터의 최대 길이(max_len) 등의 configuration을 정의해줍니다.<br/>
위 코드에서는 train_cfg의 json파일에 딕셔너리 형태로 정의된 configuration을 로드합니다.

> bert_cfg 구성<br/>
{
        "dim": 768,
	"dim_ff": 3072,
	"n_layers": 12,
	"p_drop_attn": 0.1,
	"n_heads": 12,
	"p_drop_hidden": 0.1,
	"max_len": 512,
	"n_segments": 2,
	"vocab_size": 30522
}


2) Fine-Tuning 과정에 필요한 configuration

```
model_cfg = models.Config.from_json(model_cfg)
```
BERT fine-tuning 과정은 모델을 타겟 task로 재학습시키는 과정입니다. 학습을 위해서는 seed값, bacth 크기(batch_size), 학습률(lr) 등의 하이퍼파라미터를 정의가 필요한데, 이는 model_cfg로 초기화해줍니다.

> model_cfg 구성<br/>
{
    "seed": 92,
    "batch_size": 32,
    "lr": 5e-5,
    "n_epochs": 3,
    "warmup": 0.1,
    "save_steps": 500,
    "total_steps": 6000
}

And here is some `inline code`!

### 2. DataLoader 만들기
1) Tokenizer 정의
```
tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=True)
```
FullTokenizer함수는 크게 2가지 tokenizer로 구성되어 있습니다.<br/>
하나는 주어진 텍스트 데이터를 정제하는 BasicTokenizer입니다. 문장 부호나 문자가 아닌 부분들을 제거하고, 대문자를 전부 소문자로 변경하는 과정입니다. (이때, 대문자를 그대로 유지할지, 소문자로 변경할지 여부는 미리 설정할 수 있습니다.)<br/>
다른 하나는 텍스트 데이터의 문장들을 'token'들로 변경하는 WordpieceTokenizer입니다. 대부분의 학습에 사용되는 'token'은 의미를 가지는 최소한의 언어 단위입니다. 영어에서는 주로 띄어쓰기 단위로 단어를 구분하여 token으로 사용하며, 한글에서는 형태소 단위의 token을 자주 사용합니다. BERT 모델은 사전학습에 사용된 대용량 코퍼스의 token들이 존재하기 때문에 미리 정의된 token들을 기준으로 tokenization을 합니다. 이때 대용량 코퍼스에 등장하지 않았던 단어들은 '[UNK]' 토큰으로 처리됩니다.

2) Tokenization 과정

3) DataLoader 정의


### 3. Model 정의

### 4. Train 과정

### 5. Eval 과정
