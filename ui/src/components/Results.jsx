import React from "react";
import "../Styles/Results.css";
import DownArrow from "../Vectors/downarrow.svg";
import Cart from "../Vectors/cart.png";
import Dummypro from "../Vectors/dummypro.png";

export default function Results() {
    return (
        <div className="results">
            <div className="navbar">
                <div className="logger-black">
                    Here are Suggestions for you!
                </div>
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
                    <div className="cart">
                        <img src={Cart} alt="" />
                    </div>
                </div>
            </div>
            <div className="result-content">
                <div className="card-palets">
                    <div className="upper-cards">
                        <div className="cards">
                            <div className="card-image">
                                <img src={Dummypro} alt="" />
                            </div>
                            <div className="card-info">
                                <div className="detailer">
                                    <div>Cetaphil Hydrating Moisturizer</div>
                                    <div>size - 100ml</div>
                                    <div>price - 400/-</div>
                                </div>
                                <div className="add-to-cart-div">
                                    <button className="add-to-cart">
                                        ADD TO CART
                                    </button>
                                </div>
                            </div>
                        </div>
                        <div className="cards">
                            <div className="card-image">
                                <img src={Dummypro} alt="" />
                            </div>
                            <div className="card-info">
                                <div className="detailer">
                                <div>Cetaphil Hydrating Moisturizer</div>
                                    <div>size - 100ml</div>
                                    <div>price - 400/-</div>
                                </div>
                                <div className="add-to-cart-div">
                                    <button className="add-to-cart">
                                        ADD TO CART
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div className="lower-cards">
                        <div className="cards">
                            <div className="card-image">
                                <img src={Dummypro} alt="" />
                            </div>
                            <div className="card-info">
                                <div className="detailer">
                                <div>Cetaphil Hydrating Moisturizer</div>
                                    <div>size - 100ml</div>
                                    <div>price - 400/-</div>
                                </div>
                                <div className="add-to-cart-div">
                                    <button className="add-to-cart">
                                        ADD TO CART
                                    </button>
                                </div>
                            </div>
                        </div>
                        <div className="cards">
                            <div className="card-image">
                                <img src={Dummypro} alt="" />
                            </div>
                            <div className="card-info">
                                <div className="detailer">
                                <div>Cetaphil Hydrating Moisturizer</div>
                                    <div>size - 100ml</div>
                                    <div>price - 400/-</div>
                                </div>
                                <div className="add-to-cart-div">
                                    <button className="add-to-cart">
                                        ADD TO CART
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
