#Object orientated version of creation of self attention head
block_size = 8
numberofembed = 32
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Head(nn.Module):
  #implements one head of self attention
  def __init__(self , head_size):
    super().__init__()
    self.key = nn.Linear(C , head_size , bias = False)
    self.query = nn.Linear(C , head_size, bias = False)
    self.value = nn.Linear(C , head_size , bias = False)
    self.register_buffer("tril",torch.tril(torch.ones(block_size , block_size))) #creates the triangle
    #but as the tril is not a parameter tril is just a buffer: a "temp" variable.
    #this is only needed due to us needing to hide future data

  def forward(self,x):
    B , T , C = x.shape #4 batches , 8 timecounts and 32 channels
    #4 batches
    #eahc batch has 8 tokens
    #each token has 32 numbers accossiated with it
    k = self.key(x)
    q = self.query(x)
    weight = q @ k.transpose(-2,-1) * C**-0.5  #produces a B , T ,T dimension array
    #*C**-0.5 is scaled attention
    #scaled attention is done to keep values returned from the dot product be at a reasonable range
    #this is needed as weights can get very large and we dont want certain tokens dominating in terms of communciatio, drowning the other tokens
    weight = weight.masked_fill(self.tril[:T , :T] == 0 ,float("-inf"))
    weight = F.softmax(weight, dim = -1)
    v = self.value(x)
    output = weight @ v
    return output

class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocabsize,numberofembed) #this has each idex vlaue has accossiated values in the embdded table
    self.position_embedding_table = nn.Embedding(block_size , numberofembed)
    self.languagemodelhead = nn.Linear(numberofembed , vocabsize)
    self.sa_head = Head(numberofembed)

  def forward(self, inputx , target = None ):
    B , T = inputx.shape
    token_embed = self.token_embedding_table(inputx)    #(batch , time , count)
    position_embed = self.position_embedding_table(torch.arange(T , device = device))
    x = token_embed + position_embed #with numbers accosiated to both position and identity each value now is unique and so locating and identifying is done so muhc easier
    x = self.sa_head(x)
    logits = self.languagemodelhead(x) #(batch , time , vocab) #transforms the data into the correct size to then be learnt from

    #gets the tokens , accosiated embeded table values and runs them through a linear layer
    if target == None :
      loss = None
    else:
      batchsize , tokenlength,channel = logits.shape
      logits = logits.view(batchsize*tokenlength , channel)
      #this logic also applies to the target values
      target = target.view(batchsize*tokenlength)
      loss = F.cross_entropy(logits , target) #calculates a value from the logits and targets , this is loss function whcih rates how right or wrong the guesses were

    return logits , loss

  def generate(self,idx,max_newtokens):
    #this method will be used to actually use the loss and logits by having the model generate the next values from the first value for the amount of new token wanted to be generated in that row
    for new_data in range(max_newtokens):
      idx_trim = idx[: , -block_size:]
      logits , loss = self(idx_trim)
      probabilities = F.softmax(logits , dim = -1)
      next_token = torch.multinomial(probabilities , num_samples = 1) #gets token with the highest probability
      idx = torch.cat((idx , next_token) , dim = 1) #adds that tokne with the highest probability onto x


    return idx

Bigram = BigramLanguageModel().to(device)
optimiser = torch.optim.Adam(Bigram.parameters() , lr = lr )

#now we will see if this optimiser actually work
batchsize = 32
for steps in range(25001):
  xb , yb = get_batch("train")
  logits , loss = Bigram.forward(xb , yb)
  optimiser.zero_grad() #emtpies out previous results of backward propogation so that it does not influence backward propogation
  loss.backward() #perform backward propogation to then get the amount you wantt to change the paramterrs
  optimiser.step() #performs that step to change the parameters
  if steps % 1000 == 0:
    print(f"step: {steps} loss: {loss.item()}")
