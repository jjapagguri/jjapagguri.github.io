---
layout: post
category: MODEL
---

## BERT Fine-Tuning이란?
   BERT는 대용량의 텍스트 코퍼스의 단어 임베딩을 MLM과 NSP 방식을 통해 사전학습한 transformer 모델이다. MLM과 NSP task를 잘 수행하도록 학습된 BERT는 언어의 맥락적, 문법적 특징을 이해할 수 있다고 여겨진다. 이와 같이 사전학습된 모델을 다시 해결하고자 하는 타겟 task를 잘 수행하도록 모델의 파라미터를 재조정하는 과정이 fine-tuning 과정입니다.<br/><br/> 이 글에서는 multi-class text classification을 잘하도록 BERT 파라미터를 fine-tuning하는 코드를 리뷰하겠습니다.
<br/>

## BERT Fine-tuning의 main함수 구성

### 1. Configuration 초기화
1) BERT 자체 configuration 정의
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


2) Fine-Tuning 과정에 필요한 configuration 정의

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


### 2. DataLoader 만들기
1) Tokenizer 정의
```
tokenizer = tokenization.FullTokenizer(vocab_file=vocab, do_lower_case=True)
```
FullTokenizer 객체는 크게 2가지 tokenizer로 구성되어 있습니다.<br/><br/>
하나는 주어진 텍스트 데이터를 정제하는 *BasicTokenizer*입니다. 문장 부호나 문자가 아닌 부분들을 제거하고, 대문자를 전부 소문자로 변경하는 과정입니다. (이때, 대문자를 그대로 유지할지, 소문자로 변경할지 여부는 미리 설정할 수 있습니다.)<br/><br/>
다른 하나는 텍스트 데이터의 문장들을 'token'들로 변경하는 *WordpieceTokenizer*입니다. 대부분의 학습에 사용되는 'token'은 의미를 가지는 최소한의 언어 단위입니다. 영어에서는 주로 띄어쓰기 단위로 단어를 구분하여 token으로 사용하며, 한글에서는 형태소 단위의 token을 자주 사용합니다. BERT 모델은 사전학습에 사용된 대용량 코퍼스의 token들이 존재하기 때문에 미리 정의된 token을 기준으로 tokenization을 합니다. 이때 대용량 코퍼스에 등장하지 않았던 단어는 '[UNK]' 토큰으로 처리됩니다.

2) Tokenization 과정
```
pipeline = [Tokenizing(tokenizer.convert_to_unicode, tokenizer.tokenize),
            AddSpecialTokensWithTruncation(max_len),
            TokenIndexing(tokenizer.convert_tokens_to_ids,
            TaskDataset.labels, max_len)]
dataset = TaskDataset(data_file, pipeline)
```
dataset에는 텍스트 데이터들이 (input_ids, segment_ids, input_mask, label_id)의 형태로 존재합니다. <br/>
pipeline을 통해 data_file에 있는 텍스트 데이터를 숫자 형태로 바꾸고 토큰화하는 과정을 순서대로 진행합니다.<br/>
- Tokenizing 함수 : 텍스트 데이터를 유니코드 형식으로 변경한 후, tokenization 과정을 거칩니다. --> returns (label, tokens_a, tokens_b)
- AddSpeicalTokenWithTruncation 함수 : BERT는 transformer로 구성된 모델이기 때문에 input 데이터의 길이가 일정해야 합니다. 이 함수에서는 max_len에 따라 *token_a + token_b <= max_len*이 충족되도록 데이터 일정 부분을 잘라냅니다. --> returns (label, ['[CLS]' + tokens_a + '[SEP]'], [tokens_b + '[SEP]'])
- TokenIndexing 함수 : 유니코드로 나타냈던 토큰들을 사전학습에 사용된 대용량 코퍼스 단어들의 인덱스로 변경합니다. 또한 BERT 학습에 필요한 segment_ids와 input_mask을 데이터 별로 생성합니다. --> returns (input_ids, segment_ids, input_mask, label_id)

3) DataLoader 정의

```
data_iter = DataLoader(dataset, batch_size=bert_cfg.batch_size, shuffle=True)
```
train과 eval 과정에 사용될 DataLoader를 정의합니다.

### 3. Model 정의
```
model = Classifier(model_cfg, len(TaskDataset.labels))
```
1) Classifier 구성

```
class Classifier(nn.Module):
    """ Classifier with Transformer """
    def __init__(self, cfg, n_labels):
        super().__init__()
	#BERT의 마지막 layer output 반환
        self.transformer = models.Transformer(cfg)
        self.fc = nn.Linear(cfg.dim, cfg.dim)
        self.activ = nn.Tanh()
        self.drop = nn.Dropout(cfg.p_drop_hidden)
        self.classifier = nn.Linear(cfg.dim, n_labels)

    def forward(self, input_ids, segment_ids, input_mask):
        h = self.transformer(input_ids, segment_ids, input_mask)
        #h의 0번째 토큰 임베딩, 즉 CLS 토큰 사용
        pooled_h = self.activ(self.fc(h[:, 0]))
        logits = self.classifier(self.drop(pooled_h))
        return logits
```
Text Classification Model은 BERT와 FC Layer 2개로 구성되어 있습니다.<br/>
최종 분류에는 BERT 마지막 layer의 CLS 토큰이 사용된다.

2) BERT (models.Transformer) 구성
```
class Block(nn.Module):
    """ Transformer Block """
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        self.norm1 = LayerNorm(cfg)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg)
        self.drop = nn.Dropout(cfg.p_drop_hidden)

    def forward(self, x, mask):
        h = self.attn(x, mask)
        h = self.norm1(x + self.drop(self.proj(h)))
        h = self.norm2(h + self.drop(self.pwff(h)))
        return h


class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks"""
    def __init__(self, cfg):
        super().__init__()
        self.embed = Embeddings(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])

    def forward(self, x, seg, mask):
        h = self.embed(x, seg)
        for block in self.blocks:
            h = block(h, mask)
        return h
```
Transformer(BERT)는 n_layers(주로 12개)만큼의 Block으로 이루어졌다.<br/>
이때 Block은 기본 Transformer의 encoder 구조이며, MultiHeadedSelfAttention과 PositionWiseFeedForward 과정을 거친다.<br/><br/>결과적으로 Transformer의 output으로는 input 토큰들 간의 중요도와 관련성이 반영된 토큰 임베딩들이 나오게 된다.

### 4. Train 과정
1) 손실함수와 최적화 방식 정의
```
criterion = nn.CrossEntropyLoss()
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=bert_cfg.lr,
                     warmup=bert_cfg.warmup,
                     t_total=bert_cfg.total_steps)
```
타겟 task가 Multi-Class Classification이기 때문에 CrossEntropyLoss를 손실함수로 사용한다.<br/>
또한 최적화 방식은 기존 BERT에서 사용하는 Adam Optimizer를 따른다. 이때 warmup은 파라미터 최적화가 이루어짐에 따라 학습률인 lr을 조금씩 줄여가서 loss가 최소가 되는 지점에 최대한 수렴할 수 있도록 돕는 기법이다.
2) Train 과정
```
def train(self, get_loss, pretrain_file=None):
	""" Train Loop """
        self.model.train() # train mode
        self.load(pretrain_file)
        model = self.model.to(self.device)

        global_step = 0 # global iteration steps regardless of epochs
        for e in range(self.cfg.n_epochs):
            loss_sum = 0. 
            iter_bar = tqdm(self.data_iter, desc='Iter (loss=X.XXX)')
            for i, batch in enumerate(iter_bar):
                batch = [t.to(self.device) for t in batch]

                self.optimizer.zero_grad()
		input_ids, segment_ids, input_mask, label_id = batch
		logits = model(input_ids, segment_ids, input_mask)
                loss = criterion(logits, label_id)
                loss.backward()
                self.optimizer.step()

                global_step += 1
                loss_sum += loss.item()
                iter_bar.set_description('Iter (loss=%5.3f)'%loss.item()

            print('Epoch %d/%d : Average Loss %5.3f'%(e+1, self.cfg.n_epochs, loss_sum/(i+1)))
        self.save(global_step)
```
train함수에서는
- 사전학습된 BERT의 파라미터를 pretrain_file로 **load**
- n_epochs만큼 전체 학습 데이터를 학습한다.
- 이때 batch마다 손실함수를 계산하고, 파라미터를 최적화하는 과정을 거친다.
- train 과정이 끝나면 fine-tuned된 모델의 파라미터를 **save**

### 5. Eval 과정
