

from tachles import load_mnist, show_mnist_teaser
from keras.layers import Input, Flatten, Dense, Conv2D
from keras.models import Model

show_mnist_teaser()
(x_train, y_train), (x_test, y_test) = load_mnist()

print 'x shape: ', x_train.shape, ' y: ', y_train.shape


inp = Input((28, 28, 1))

layer1 = Conv2D(64, 3, 3)(inp)
layer2 = Flatten()(layer1)
layer3 = Dense(10, activation = 'softmax')(layer2)

model = Model(inp, layer3)
model.summary()
model.compile(optimizer='sgd', loss='categorical_crossentropy')
model.fit(x_train, y_train)