from keras.models import load_model

model = load_model('my_model.h5')

xVal = ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
# evaluate loaded model on test data
# Define X_test & Y_test data first
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
#               metrics=['accuracy'])
# score = model.evaluate(X_test, Y_test, verbose=0)


yFit = model.predict(xVal, batch_size=10, verbose=1)
print()
print(yFit)

