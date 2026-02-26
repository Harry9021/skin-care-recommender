import React, { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import "../Styles/Home.css";
import "../Styles/Navbar.css";
import DownArrow from "../Vectors/downarrow.svg";
import Cart from "../Vectors/cart.png";
import image1 from "../Vectors/image1.png";
import image2 from "../Vectors/image2.png";
import image3 from "../Vectors/image3.png";
import image4 from "../Vectors/image4.png";
import { GoogleOAuthProvider, GoogleLogin } from "@react-oauth/google";

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";
const GOOGLE_CLIENT_ID = process.env.REACT_APP_GOOGLE_CLIENT_ID || "";
const AUTH_TOKEN_KEY = "authToken";
const AUTH_USER_KEY = "authUser";

export default function Home() {
    const navigate = useNavigate();
    const [dropdownOpen, setDropdownOpen] = useState(false);
    const [authError, setAuthError] = useState("");
    const [isAuthLoading, setIsAuthLoading] = useState(false);
    const [pendingTryNow, setPendingTryNow] = useState(false);
    const [user, setUser] = useState(null);

    useEffect(() => {
        const storedUser = localStorage.getItem(AUTH_USER_KEY);
        if (storedUser) {
            try {
                setUser(JSON.parse(storedUser));
            } catch (e) {
                localStorage.removeItem(AUTH_USER_KEY);
            }
        }
    }, []);

    const isLoggedIn = !!localStorage.getItem(AUTH_TOKEN_KEY);

    const handleOAuthSuccess = async (response) => {
        try {
            setAuthError("");
            setIsAuthLoading(true);

            const googleToken = response?.credential;
            if (!googleToken) {
                throw new Error("Google credential missing");
            }

            const authResponse = await fetch(`${API_BASE_URL}/api/auth/google`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ token: googleToken })
            });

            const data = await authResponse.json();
            if (!authResponse.ok || !data.token) {
                throw new Error(data.message || "OAuth login failed");
            }

            localStorage.setItem(AUTH_TOKEN_KEY, data.token);
            localStorage.setItem(AUTH_USER_KEY, JSON.stringify(data.user || {}));
            setUser(data.user || null);
            setDropdownOpen(false);

            if (pendingTryNow) {
                setPendingTryNow(false);
                navigate("/form");
            }
        } catch (error) {
            setAuthError(error.message || "Login failed. Please try again.");
        } finally {
            setIsAuthLoading(false);
        }
    };

    const handleOAuthFailure = () => {
        setAuthError("Google login was cancelled or failed.");
    };

    const handleTryNowClick = () => {
        if (isLoggedIn) {
            navigate("/form");
            return;
        }
        setPendingTryNow(true);
        setDropdownOpen(true);
        setAuthError("Please login with Google to continue.");
    };

    const handleLogout = () => {
        localStorage.removeItem(AUTH_TOKEN_KEY);
        localStorage.removeItem(AUTH_USER_KEY);
        setUser(null);
        setPendingTryNow(false);
        setAuthError("");
    };

    return (
        <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID || "YOUR_GOOGLE_CLIENT_ID"}>
            <div className="home">
                <div className="navbar">
                    <div className="logger-black"></div>
                    <div className="logo">SKINCARE</div>
                    <div className="logger">
                        <div
                            className="logger-1"
                            onClick={() => setDropdownOpen(!dropdownOpen)}
                        >
                            <div className="loginButton">
                                <div style={{ fontSize: 20 }}>
                                    {isLoggedIn ? "Logged In" : "Login with Google"}
                                </div>
                                <div style={{ fontSize: 24 }}>
                                    {user?.name || "My Account"}
                                </div>
                            </div>
                            <div className="arrow">
                                <img src={DownArrow} alt="dropdown arrow" />
                            </div>
                        </div>
                        {dropdownOpen && (
                            <div className="popup-menu">
                                <div className="popup-login">
                                    <h2
                                        style={{
                                            fontFamily: "Satisfy, cursive",
                                            fontSize: "24px",
                                        }}
                                    >
                                        {isLoggedIn ? "Welcome" : "Sign in with Google"}
                                    </h2>
                                    {authError && (
                                        <p style={{ color: "#c62828", margin: "8px 0", textAlign: "center" }}>
                                            {authError}
                                        </p>
                                    )}
                                    {!isLoggedIn ? (
                                        <div className="google-auth">
                                            <GoogleLogin
                                                onSuccess={handleOAuthSuccess}
                                                onError={handleOAuthFailure}
                                            />
                                        </div>
                                    ) : (
                                        <button className="login-button" onClick={handleLogout}>
                                            Logout
                                        </button>
                                    )}
                                    {isAuthLoading && (
                                        <p style={{ marginTop: "12px", textAlign: "center" }}>
                                            Signing you in...
                                        </p>
                                    )}
                                </div>
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
                            <button className="try-button" onClick={handleTryNowClick}>Try Now</button>
                        </div>
                    </div>
                </div>
            </div>
        </GoogleOAuthProvider>
    );
}
