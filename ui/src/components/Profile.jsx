import React, { useState } from "react";

const UserProfile = () => {
  const [isEditing, setIsEditing] = useState({
    username: false,
    email: false,
    address: false,
    nickname: false,
    dob: false,
  });

  const [profile, setProfile] = useState({
    username: "Jenny Wilson",
    email: "jenny@gmail.com",
    address: "New York, USA",
    nickname: "Sky Angel",
    dob: "April 28, 1981",
  });

  const handleEditClick = (field) => {
    setIsEditing((prevState) => ({
      ...prevState,
      [field]: !prevState[field],
    }));
  };

  const handleChange = (e, field) => {
    setProfile((prevState) => ({
      ...prevState,
      [field]: e.target.value,
    }));
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100">
      <div className="bg-white shadow-md rounded-lg w-96 p-6">
        <div className="flex flex-col items-center">
          {/* Profile Picture */}
          <img
            src="https://randomuser.me/api/portraits/women/79.jpg"
            alt="Profile"
            className="w-24 h-24 rounded-full object-cover mb-4"
          />
          {/* Name and Username */}
          <h2 className="text-2xl font-semibold">Jessica Alba</h2>
          <p className="text-gray-500">@jennywilson</p>
        </div>

        {/* Profile Details */}
        <div className="mt-6">
          <ProfileItem
            label="Username"
            value={profile.username}
            isEditing={isEditing.username}
            onEditClick={() => handleEditClick("username")}
            onChange={(e) => handleChange(e, "username")}
          />
          <ProfileItem
            label="Email"
            value={profile.email}
            isEditing={isEditing.email}
            onEditClick={() => handleEditClick("email")}
            onChange={(e) => handleChange(e, "email")}
          />
          <ProfileItem
            label="Address"
            value={profile.address}
            isEditing={isEditing.address}
            onEditClick={() => handleEditClick("address")}
            onChange={(e) => handleChange(e, "address")}
          />
          <ProfileItem
            label="Nickname"
            value={profile.nickname}
            isEditing={isEditing.nickname}
            onEditClick={() => handleEditClick("nickname")}
            onChange={(e) => handleChange(e, "nickname")}
          />
          <ProfileItem
            label="DOB"
            value={profile.dob}
            isEditing={isEditing.dob}
            onEditClick={() => handleEditClick("dob")}
            onChange={(e) => handleChange(e, "dob")}
          />
        </div>
      </div>
    </div>
  );
};

const ProfileItem = ({ label, value, isEditing, onEditClick, onChange }) => {
  return (
    <div className="flex justify-between items-center border-b py-2">
      <div>
        <span className="text-gray-600">{label}</span>
        {isEditing ? (
          <input
            type="text"
            value={value}
            onChange={onChange}
            className="font-medium text-gray-800 border border-gray-300 rounded px-2 py-1 mt-1"
          />
        ) : (
          <p className="font-medium text-gray-800">{value}</p>
        )}
      </div>
      <button className="text-blue-500 hover:underline" onClick={onEditClick}>
        <svg
          xmlns="http://www.w3.org/2000/svg"
          fill="none"
          viewBox="0 0 24 24"
          strokeWidth={1.5}
          stroke="currentColor"
          className="w-5 h-5"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M15.232 5.232a3 3 0 114.243 4.243L10.242 18.707a4.5 4.5 0 01-1.864 1.132l-3.285.987.987-3.285a4.5 4.5 0 011.132-1.864l9.233-9.232z"
          />
        </svg>
      </button>
    </div>
  );
};

export default UserProfile;
