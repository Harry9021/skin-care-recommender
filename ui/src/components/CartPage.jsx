import React from "react";
import DownArrow from "../Vectors/downarrow.svg";
import Cart from "../Vectors/cart.png";
import "../Styles/CartPage.css"; // Make sure to create this file

const CartPage = () => {
    return (
        <div className="shopping-container">
            {/* Navbar */}
            <div className="navbar">
                <div className="logger-black">Shopping Cart</div>
                <div className="logger">
                    <div className="logger-1">
                        <div className="loginButton">
                            <div style={{ fontSizes: 20 }}>Login/Signup</div>
                            <div style={{ fontSize: 24 }}>My Account</div>
                        </div>
                        <div className="arrow">
                            <img src={DownArrow} alt="" />
                        </div>
                    </div>
                    {/* <div className="cart">
                        <img src={Cart} alt="" />
                    </div> */}
                </div>
            </div>

            {/* Main Content */}
            <div className="main-content">
                {/* Shopping Cart Section */}
                <div className="shopping-cart">
                    <h2>Shopping Cart</h2>
                    <div className="cart-items">
                        <CartItem
                            img="https://www.example.com/moisturizer.jpg"
                            name="Moisturizer"
                            size="150ml"
                            price="300/-"
                        />
                        <CartItem
                            img="https://www.example.com/sunscreen.jpg"
                            name="Sunscreen"
                            size="100ml"
                            price="400/-"
                        />
                        <CartItem
                            img="https://www.example.com/lipbalm.jpg"
                            name="Lip Balm"
                            size="5gm"
                            price="100/-"
                        />
                    </div>
                </div>

                {/* Summary Section */}
                <div className="summary">
                    <h2>Summary</h2>
                    <p>Items: 3</p>
                    <p className="total-price">Total Price: 800 Rs</p>

                    <div className="summary-inputs">
                        <div>
                            <label>Address</label>
                            <input type="text" placeholder="Enter here" />
                        </div>

                        <div>
                            <label>PIN CODE</label>
                            <input type="text" placeholder="Value" />
                        </div>

                        <div>
                            <label>Payment Mode</label>
                            <select>
                                <option>COD</option>
                                <option>Credit Card</option>
                                <option>Debit Card</option>
                                <option>UPI</option>
                            </select>
                        </div>

                        <button className="order-button">Order Now!</button>
                    </div>
                </div>
            </div>
        </div>
    );
};

const CartItem = ({ img, name, size, price }) => {
    return (
        <div className="cart-item">
            <img src={img} alt={name} className="item-image" />
            <div className="item-details">
                <h3>{name}</h3>
                <p>Size: {size}</p>
                <p>Price: {price}</p>
            </div>
        </div>
    );
};

export default CartPage;
