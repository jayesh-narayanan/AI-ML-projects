import torch

class Encoder:
    def __init__(self,input_size=128, hidden_size=512, num_layers=2):
        self.rnn = torch.nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
    def forward(self, x):
        # Padding of input sequence
        
        output, hidden = self.rnn(x)
        return output, hidden

class Decoder:
    def __init__(self, hidden_size=512, output_size=128, num_layers=2):
        self.rnn = torch.nn.RNN(hidden_size, output_size, num_layers, batch_first=True)
    def forward(self, x, hidden):
        output, hidden = self.rnn(x, hidden)
        return output, hidden
    
class Seq2Seq:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, src, trg):
        encoder_output, hidden = self.encoder.forward(src)
        decoder_output, hidden = self.decoder.forward(trg, hidden)
        return decoder_output
    