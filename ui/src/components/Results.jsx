import React, { useContext } from "react";
import "../Styles/Results.css";
import DownArrow from "../Vectors/downarrow.svg";
import Cart from "../Vectors/cart.png";
import { Link, useLocation, useNavigate } from "react-router-dom";
import Resultcard from "./cards/Resultcard";
import { CartContext } from "./context/CartContext"; // Import CartContext

export default function Results() {
    const location = useLocation();
    const navigate = useNavigate();
    const { addToCart } = useContext(CartContext); // Access addToCart function

    const recommendations = location.state?.recommendations || {}; // Default to an empty object

    let items = [];
    if (recommendations.data) {
        try {
            items = JSON.parse(recommendations.data); // Parse the JSON string
        } catch (error) {
            console.error("Error parsing recommendations data:", error);
        }
    }

    return (
        <div className="results">
            <div className="navbar">
                <div className="logger-black">
                    Here are Suggestions for you!
                </div>
                <div className="logger">
                    <div className="logger-1">
                        <div className="loginButton">
                            <div style={{ fontSize: 20 }}>Login/Signup</div>
                            <div style={{ fontSize: 24 }}>My Account</div>
                        </div>
                        <div className="arrow">
                            <img src={DownArrow} alt="Down Arrow" />
                        </div>
                    </div>
                    <Link to="/cart">
                        <img
                            src={Cart}
                            alt="Cart"
                            style={{ height: "40px", width: "40px" }}
                        />
                    </Link>
                </div>
            </div>

            <button onClick={() => navigate(-1)} className="back-button">
                Back
            </button>

            <div className="result-content">
                {items.length > 0 ? (
                    items.map((item, index) => (
                        <Resultcard
                            key={index}
                            item={item}
                            recommendationNumber={index + 1}
                            addToCart={addToCart} // Pass the addToCart function
                        />
                    ))
                ) : (
                    <div>No recommendations available.</div>
                )}
            </div>
        </div>
    );
}
