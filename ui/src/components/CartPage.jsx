import React, { useContext, useState } from "react";
import { useNavigate } from "react-router-dom"; // Import useNavigate for navigation
import "../Styles/CartPage.css";
import { CartContext } from "./context/CartContext"; // Import CartContext

const CartPage = () => {
    const { cartItems, clearCart } = useContext(CartContext); // Access cartItems and clearCart from the context
    const [orderMessage, setOrderMessage] = useState(""); // State for order message
    const [address, setAddress] = useState(""); // State for address
    const [pinCode, setPinCode] = useState(""); // State for PIN code
    const [paymentMode, setPaymentMode] = useState("COD"); // State for payment mode
    const navigate = useNavigate(); // Initialize navigate

    // Calculate the total price, ensuring item.price is treated as a number
    const totalPrice = cartItems.reduce((total, item) => {
        const price = Number(item.price); // Ensure price is a number
        return total + price; // Add to total
    }, 0);

    // Handle order placement
    const handleOrderNow = () => {
        // Check if all fields are filled
        if (!address || !pinCode || !paymentMode) {
            setOrderMessage("Please fill all the required fields.");
            return; // Stop further execution
        }

        if (cartItems.length > 0) {
            setOrderMessage("Order placed successfully!");
            clearCart(); // Clear the cart items
        } else {
            setOrderMessage("Your cart is empty.");
        }

        // Clear the message after a few seconds
        setTimeout(() => {
            setOrderMessage("");
        }, 3000);
    };

    // Handle continue shopping button click
    const handleContinueShopping = () => {
        navigate("/form"); // Redirect to the form page
    };

    return (
        <div className="shopping-container">
            {/* Navbar */}
            <div className="navbar">
                <div className="logger-black">Shopping Cart</div>
            </div>

            {/* Main Content */}
            <div className="main-content">
                {/* Shopping Cart Section */}
                <div className="shopping-cart">
                    <h2>Shopping Cart</h2>
                    <div className="cart-items">
                        {cartItems.length > 0 ? (
                            cartItems.map((item, index) => (
                                <CartItem key={index} {...item} /> // Use CartItem to display each item
                            ))
                        ) : (
                            <div>Your cart is empty.</div>
                        )}
                    </div>
                </div>

                {/* Summary Section */}
                <div className="summary">
                    <h2>Summary</h2>
                    <p>Items: {cartItems.length}</p>
                    <p className="total-price">
                        Total Price: ₹ {totalPrice.toLocaleString()} {/* Format total price with commas */}
                    </p>

                    <div className="summary-inputs">
                        <div>
                            <label>Address</label>
                            <input
                                type="text"
                                placeholder="Enter here"
                                value={address} // Set value from state
                                onChange={(e) => setAddress(e.target.value)} // Update state on input change
                            />
                        </div>

                        <div>
                            <label>PIN CODE</label>
                            <input
                                type="text"
                                placeholder="Value"
                                value={pinCode} // Set value from state
                                onChange={(e) => setPinCode(e.target.value)} // Update state on input change
                            />
                        </div>

                        <div>
                            <label>Payment Mode</label>
                            <select
                                value={paymentMode} // Set value from state
                                onChange={(e) => setPaymentMode(e.target.value)} // Update state on select change
                            >
                                <option value="COD">COD</option>
                                <option value="Credit Card">Credit Card</option>
                                <option value="Debit Card">Debit Card</option>
                                <option value="UPI">UPI</option>
                            </select>
                        </div>

                        <button className="order-button" onClick={handleOrderNow}>
                            Order Now!
                        </button>
                    </div>
                </div>
            </div>

            {/* Continue Shopping Button */}
            <div className="continue-shopping-container">
                <button className="continue-shopping-button" onClick={handleContinueShopping}>
                    Continue Shopping
                </button>
            </div>

            {/* Order Message Popup */}
            {orderMessage && (
                <div className="order-popup">
                    {orderMessage}
                </div>
            )}
        </div>
    );
};

const CartItem = ({ img, name, size, price }) => (
    <div className="cart-item">
        <img src={img} alt={name} className="item-image" />
        <div className="item-details">
            <h3>{name}</h3>
            <p>Size: {size}</p>
            <p>Price: ₹ {Number(price).toLocaleString()}</p> {/* Format the price with commas */}
        </div>
    </div>
);

export default CartPage;
