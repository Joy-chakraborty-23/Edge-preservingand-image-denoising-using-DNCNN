import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

def train_model(model, X, y, batch_size=128, epochs=50):
    # Class balancing
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights = {i: w for i, w in enumerate(weights)}
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, decay=5e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        class_weight=class_weights
    )
    return model, history
