import React, { useContext } from "react";
import "../Styles/Results.css";
import Cart from "../Vectors/cart.png";
import { Link, useLocation, useNavigate } from "react-router-dom";
import Resultcard from "./cards/Resultcard";
import { CartContext } from "./context/CartContext";
import { IoMdArrowRoundBack } from "react-icons/io";

export default function Results() {
    const location = useLocation();
    const navigate = useNavigate();
    const { addToCart } = useContext(CartContext);

    const recommendations = location.state?.recommendations || [];
    const userInput = location.state?.userInput || {};

    const handleBackClick = () => {
        navigate(-1);
    };

    return (
        <div className="results-container">
            {/* Header */}
            <header className="results-header">
                <div className="header-content">
                    <div className="header-left">
                        <button className="back-button" onClick={handleBackClick} title="Go back">
                            <IoMdArrowRoundBack size={24} />
                        </button>
                        <h1>Personalized Recommendations</h1>
                    </div>
                    <div className="header-right">
                        <div className="user-info">
                            <span className="user-label">Your Profile</span>
                            <div className="user-details">
                                <small>Skin Type: <strong>{userInput.skinType}</strong></small>
                            </div>
                        </div>
                        <Link to="/cart" className="cart-link" title="View cart">
                            <img src={Cart} alt="Shopping Cart" className="cart-icon" />
                            <span className="cart-badge">Cart</span>
                        </Link>
                    </div>
                </div>
            </header>

            {/* Results Content */}
            <main className="results-main">
                <div className="results-info">
                    <h2>Top {recommendations.length} Recommended Products</h2>
                    <p className="results-subtitle">
                        Curated based on your skin type and concerns
                    </p>
                </div>

                {recommendations.length > 0 ? (
                    <div className="results-grid">
                        {recommendations.map((item, index) => (
                            <div key={index} className="result-item">
                                <Resultcard
                                    item={item}
                                    recommendationNumber={index + 1}
                                    addToCart={addToCart}
                                />
                            </div>
                        ))}
                    </div>
                ) : (
                    <div className="no-results">
                        <p>No recommendations available. Please try again.</p>
                        <button onClick={handleBackClick} className="retry-button">
                            Go Back to Form
                        </button>
                    </div>
                )}
            </main>

            {/* Footer */}
            <footer className="results-footer">
                <p>&copy; 2024 Skincare Recommendation System. All rights reserved.</p>
            </footer>
        </div>
    );
}
