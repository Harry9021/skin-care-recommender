// context/CartContext.js
import React, { createContext, useState } from 'react';

export const CartContext = createContext();

export const CartProvider = ({ children }) => {
    const [cartItems, setCartItems] = useState([]);

    const addToCart = (item) => {
        setCartItems((prevItems) => [...prevItems, item]); // Add the new item to the cart
    };

    // Function to clear the cart
    const clearCart = () => {
        setCartItems([]); // Set cartItems to an empty array
    };



    // Other cart-related functions can be added here
    // For example: addItem, removeItem, etc.

    return (
        <CartContext.Provider value={{ cartItems, setCartItems, clearCart, addToCart }}>
            {children}
        </CartContext.Provider>
    );
};
