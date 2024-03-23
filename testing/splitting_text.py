import spacy 
nlp = spacy.load('en_core_web_sm')

#text_a = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Cras eu elementum augue. Praesent sed mi lorem. In pharetra aliquam lacinia. In mattis metus odio, quis placerat urna tincidunt sit amet. Aenean condimentum nisl vel semper pharetra. Pellentesque sed tortor sem. Proin molestie et leo vel ornare. Nullam feugiat laoreet massa sed dignissim. Praesent aliquam elit ligula, a mollis velit faucibus a. Suspendisse ipsum mauris, feugiat sed urna ac, placerat molestie arcu. Integer erat turpis, aliquet vel augue eu, tristique aliquet urna. Fusce hendrerit commodo ante, ullamcorper ornare justo convallis vitae. Etiam placerat erat sit amet vestibulum lobortis. Nunc at quam a turpis consequat varius. Aenean ac leo consectetur justo semper dictum. Morbi ullamcorper quis sapien in fermentum. '
#text_b = 'Vivamus fringilla nunc ut odio maximus, a gravida nulla porta. Proin eu ullamcorper enim. Nam vitae velit dictum, blandit lectus in, lacinia metus. Fusce vulputate dapibus sapien, sit amet lacinia dolor bibendum eu. Integer rutrum finibus eleifend. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Integer iaculis tincidunt tellus, sit amet porta leo accumsan non. Cras eu finibus ante. Aenean vehicula metus sit amet ante commodo pretium. '
#text_c = 'Proin semper lacinia dui ut cursus. Sed sed nunc viverra, fermentum tortor non, cursus libero. Cras fringilla, lorem mollis imperdiet scelerisque, erat quam imperdiet velit, eu suscipit velit libero sed tortor. Curabitur nec dignissim ipsum. Integer in vestibulum mauris. Cras et orci ultricies, ultrices ligula nec, varius ligula. In facilisis odio ut odio pellentesque pharetra. In ante est, ultricies a nisl at, congue luctus arcu. Pellentesque egestas ipsum ex, efficitur ornare dui pellentesque id. Quisque interdum elementum ligula vel sollicitudin. Pellentesque vitae neque eu ex facilisis dapibus a non sapien. '


x_train = [ ['1A.2A.3A.4A.5A.6A', '1B.2B.3B.4B.5B.6B'],
            ['1C.2C.3C.4C.5C.6C','1D.2D.3D.4D.5D.6D']]

def sentences(text, n):
    sents = text.split(".")
    return [' '.join(sents[i:i+n]) for i in range(0, len(sents), n)]

def split_text(x_train, y_train,sentence_length=3):
    x_out = []
    y_out = []
    
    for i,pair in enumerate(x_train):
        kt = pair[0]
        ut = pair[1]
        s_ut = sentences(ut, sentence_length)
        s_kt = sentences(kt, sentence_length)

        for sentence_kt in s_kt:
            for sentence_ut in s_ut:
                x_out.append([sentence_kt, sentence_ut])
                y_out.append(y_train[i])
        
    return x_out, y_out

y_train = [0,1]

x_train,y_train = split_text(x_train, y_train,sentence_length=1)
print(x_train)
print(y_train)



