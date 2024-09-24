import React, { useState } from "react";
import { Link } from "react-router-dom";
import "../Styles/Home.css";
import "../Styles/Navbar.css";
import DownArrow from "../Vectors/downarrow.svg";
import Cart from "../Vectors/cart.png";
import image1 from "../Vectors/image1.png";
import image2 from "../Vectors/image2.png";
import image3 from "../Vectors/image3.png";
import image4 from "../Vectors/image4.png";
import { GoogleOAuthProvider, GoogleLogin } from '@react-oauth/google';

export default function Home() {
    const [dropdownOpen, setDropdownOpen] = useState(false);
    const [isLogin, setIsLogin] = useState(true); // Toggle between Login and Signup

    const handleLoginSuccess = (response) => {
        console.log("Login Success:", response);
    };

    const handleLoginFailure = (response) => {
        console.error("Login Failed:", response);
    };

    return (
        <GoogleOAuthProvider clientId="YOUR_GOOGLE_CLIENT_ID">
            <div className="home">
                <div className="navbar">
                    <div className="logger-black"></div>
                    <div className="logo">SKINCARE</div>
                    <div className="logger">
                        <div className="logger-1" onClick={() => setDropdownOpen(!dropdownOpen)}>
                            <div className="loginButton">
                                <div style={{ fontSize: 20 }}>Login/Signup</div>
                                <div style={{ fontSize: 24 }}>My Account</div>
                            </div>
                            <div className="arrow">
                                <img src={DownArrow} alt="dropdown arrow" />
                            </div>
                        </div>
                        {dropdownOpen && (
                            <div className="popup-menu">
                                {isLogin ? (
                                    <div className="popup-login">
                                        <h2>Welcome Back!</h2>
                                        <input type="email" placeholder="Email" />
                                        <input type="password" placeholder="Password" />
                                        <button className="login-button">Login</button>
                                        <p>New customer? <span onClick={() => setIsLogin(false)}>Create your account</span></p>
                                        <div className="google-auth">
                                            <GoogleLogin
                                                onSuccess={handleLoginSuccess}
                                                onError={handleLoginFailure}
                                            />
                                        </div>
                                    </div>
                                ) : (
                                    <div className="popup-signup">
                                        <h2>Welcome!</h2>
                                        <input type="text" placeholder="Name" />
                                        <input type="text" placeholder="Gender" />
                                        <input type="email" placeholder="Email" />
                                        <input type="password" placeholder="New Password" />
                                        <input type="password" placeholder="Confirm Password" />
                                        <button className="signup-button">Sign Up</button>
                                        <p>Already have an account? <span onClick={() => setIsLogin(true)}>Login</span></p>
                                    </div>
                                )}
                            </div>
                        )}
                        <Link to="/cart">
                            <div className="cart">
                                <img src={Cart} alt="cart" />
                            </div>
                        </Link>
                    </div>
                </div>

                <div className="home-content">
                    <div className="imager">
                        <img className="image-1" src={image1} alt="" />
                        <img className="image-2" src={image2} alt="" />
                        <img className="image-3" src={image3} alt="" />
                        <img className="image-4" src={image4} alt="" />
                    </div>
                    <div className="written-content">
                        <div className="written">
                            Here’s How You Can Determine Your Skin Type. Knowing
                            your skin type is essential for creating a skincare
                            routine that works for you.
                        </div>
                        <div className="try-button-div">
                            <Link to="/form">
                                <button className="try-button">Try Now</button>
                            </Link>
                        </div>
                    </div>
                </div>
            </div>
        </GoogleOAuthProvider>
    );
}
