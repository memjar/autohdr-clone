'use client';

import React, { useState } from 'react';
import Link from 'next/link';

// Icons
const Zap = ({ className }: { className?: string }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
  </svg>
);

const Lock = ({ className }: { className?: string }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
    <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
    <path d="M7 11V7a5 5 0 0 1 10 0v4" />
  </svg>
);

const Users = ({ className }: { className?: string }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
    <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
    <circle cx="9" cy="7" r="4" />
    <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
    <path d="M16 3.13a4 4 0 0 1 0 7.75" />
  </svg>
);

const Cloud = ({ className }: { className?: string }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
    <path d="M18 10h-1.26A8 8 0 1 0 9 20h9a5 5 0 0 0 0-10z" />
  </svg>
);

const Gift = ({ className }: { className?: string }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
    <polyline points="20 12 20 22 4 22 4 12" />
    <rect x="2" y="7" width="20" height="5" />
    <line x1="12" y1="22" x2="12" y2="7" />
    <path d="M12 7H7.5a2.5 2.5 0 0 1 0-5C11 2 12 7 12 7z" />
    <path d="M12 7h4.5a2.5 2.5 0 0 0 0-5C13 2 12 7 12 7z" />
  </svg>
);

const LogOut = ({ className }: { className?: string }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
    <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
    <polyline points="16 17 21 12 16 7" />
    <line x1="21" y1="12" x2="9" y2="12" />
  </svg>
);

const ArrowLeft = ({ className }: { className?: string }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
    <line x1="19" y1="12" x2="5" y2="12" />
    <polyline points="12 19 5 12 12 5" />
  </svg>
);

const Copy = ({ className }: { className?: string }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
    <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
  </svg>
);

const Mail = ({ className }: { className?: string }) => (
  <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
    <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z" />
    <polyline points="22,6 12,13 2,6" />
  </svg>
);

interface StyleSettings {
  interiorExterior: 'interior' | 'exterior';
  brightness: number;
  contrast: number;
  vibrance: number;
  whiteBalance: number;
  windowPullIntensity: string;
  cloudStyle: string;
  twilightStyle: string;
  tvScreenReplacement: string;
  fireInFireplace: boolean;
  declutter: boolean;
  interiorClouds: boolean;
  exteriorClouds: boolean;
  deduplication: boolean;
  walkthroughReorder: boolean;
  retainOriginalSky: boolean;
  perspectiveCorrection: boolean;
  grassReplacement: boolean;
  signRemoval: boolean;
}

const defaultSettings: StyleSettings = {
  interiorExterior: 'interior',
  brightness: 0,
  contrast: 0,
  vibrance: 0,
  whiteBalance: 0,
  windowPullIntensity: 'natural',
  cloudStyle: 'fluffy-whispy',
  twilightStyle: 'pink',
  tvScreenReplacement: 'none',
  fireInFireplace: false,
  declutter: true,
  interiorClouds: true,
  exteriorClouds: true,
  deduplication: false,
  walkthroughReorder: false,
  retainOriginalSky: false,
  perspectiveCorrection: true,
  grassReplacement: false,
  signRemoval: false
};

type TabId = 'style-settings' | 'account' | 'team' | 'automation' | 'affiliate';

const dashboardTabs = [
  { id: 'style-settings' as TabId, label: 'Style Settings', icon: Zap },
  { id: 'account' as TabId, label: 'Account & Subscription', icon: Lock },
  { id: 'team' as TabId, label: 'Add Your Team', icon: Users },
  { id: 'automation' as TabId, label: 'Dropbox Automation', icon: Cloud },
  { id: 'affiliate' as TabId, label: 'Affiliate Program', icon: Gift }
];

export default function DashboardPage() {
  const [activeTab, setActiveTab] = useState<TabId>('style-settings');
  const [settings, setSettings] = useState<StyleSettings>(defaultSettings);

  return (
    <div className="min-h-screen bg-black text-white pb-24">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-4">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
          <div className="flex items-center gap-6">
            <Link
              href="/"
              className="flex items-center gap-2 text-gray-400 hover:text-cyan-400 transition"
            >
              <ArrowLeft className="w-5 h-5" />
              Back to Editor
            </Link>
            <div>
              <h1 className="text-2xl font-bold">Dashboard</h1>
              <p className="text-gray-400 text-sm">Manage your account and settings</p>
            </div>
          </div>
          <button className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg transition">
            <LogOut className="w-4 h-4" />
            Sign Out
          </button>
        </div>
      </header>

      <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-4 gap-6 p-6">
        {/* Sidebar Navigation */}
        <div className="md:col-span-1">
          <nav className="space-y-2 sticky top-6">
            {dashboardTabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`w-full text-left px-4 py-3 rounded-lg flex items-center gap-3 transition ${
                  activeTab === tab.id
                    ? 'bg-cyan-400 bg-opacity-20 text-cyan-400 border-l-2 border-cyan-400'
                    : 'text-gray-300 hover:bg-gray-900'
                }`}
              >
                <tab.icon className="w-5 h-5" />
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        {/* Main Content */}
        <div className="md:col-span-3">
          {activeTab === 'style-settings' && (
            <StyleSettingsPanel settings={settings} setSettings={setSettings} />
          )}
          {activeTab === 'account' && <AccountPanel />}
          {activeTab === 'team' && <TeamPanel />}
          {activeTab === 'automation' && <AutomationPanel />}
          {activeTab === 'affiliate' && <AffiliatePanel />}
        </div>
      </div>
    </div>
  );
}

// Style Settings Panel
function StyleSettingsPanel({
  settings,
  setSettings
}: {
  settings: StyleSettings;
  setSettings: React.Dispatch<React.SetStateAction<StyleSettings>>;
}) {
  const sliders = [
    { label: 'Brightness', key: 'brightness' as const },
    { label: 'Contrast', key: 'contrast' as const },
    { label: 'Vibrance', key: 'vibrance' as const },
    { label: 'White Balance', key: 'whiteBalance' as const }
  ];

  const toggles = [
    { label: 'Fire in Fireplace', key: 'fireInFireplace' as const },
    { label: 'Declutter', key: 'declutter' as const },
    { label: 'Interior Clouds', key: 'interiorClouds' as const },
    { label: 'Exterior Clouds', key: 'exteriorClouds' as const },
    { label: 'Deduplication', key: 'deduplication' as const },
    { label: 'Walkthrough Reorder', key: 'walkthroughReorder' as const },
    { label: 'Retain Original Sky', key: 'retainOriginalSky' as const },
    { label: 'Perspective Correction', key: 'perspectiveCorrection' as const },
    { label: 'Grass Replacement', key: 'grassReplacement' as const },
    { label: 'Sign Removal', key: 'signRemoval' as const }
  ];

  return (
    <div className="space-y-8">
      <div className="bg-gray-950 border border-gray-800 rounded-xl overflow-hidden">
        {/* Preview Image */}
        <div className="h-64 bg-gradient-to-br from-gray-800 to-gray-900 flex items-center justify-center">
          <p className="text-gray-500">Preview image will appear here</p>
        </div>

        <div className="p-8">
          <h3 className="text-2xl font-bold mb-6">Style Settings</h3>

          {/* Interior/Exterior Toggle */}
          <div className="flex gap-4 mb-8">
            <button
              onClick={() => setSettings({ ...settings, interiorExterior: 'interior' })}
              className={`px-6 py-2 rounded-lg font-semibold transition ${
                settings.interiorExterior === 'interior'
                  ? 'bg-cyan-400 text-black'
                  : 'bg-gray-800 text-white hover:bg-gray-700'
              }`}
            >
              Interior
            </button>
            <button
              onClick={() => setSettings({ ...settings, interiorExterior: 'exterior' })}
              className={`px-6 py-2 rounded-lg font-semibold transition ${
                settings.interiorExterior === 'exterior'
                  ? 'bg-cyan-400 text-black'
                  : 'bg-gray-800 text-white hover:bg-gray-700'
              }`}
            >
              Exterior
            </button>
          </div>

          {/* Sliders Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
            {sliders.map((slider) => (
              <div key={slider.key}>
                <div className="flex justify-between mb-2">
                  <label className="font-semibold">{slider.label}</label>
                  <span className="text-gray-400 text-sm">
                    {settings[slider.key]} {settings[slider.key] === 0 && '(recommended)'}
                  </span>
                </div>
                <input
                  type="range"
                  min="-2"
                  max="2"
                  step="0.1"
                  value={settings[slider.key]}
                  onChange={(e) =>
                    setSettings({
                      ...settings,
                      [slider.key]: parseFloat(e.target.value)
                    })
                  }
                  className="w-full accent-cyan-400 cursor-pointer"
                />
              </div>
            ))}
          </div>

          {/* Dropdowns */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <div>
              <label className="block font-semibold mb-2">Window Pull Intensity</label>
              <select
                value={settings.windowPullIntensity}
                onChange={(e) => setSettings({ ...settings, windowPullIntensity: e.target.value })}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-white hover:border-gray-600 focus:border-cyan-400 outline-none"
              >
                <option value="natural">Natural</option>
                <option value="strong">Strong</option>
                <option value="subtle">Subtle</option>
              </select>
            </div>

            <div>
              <label className="block font-semibold mb-2">Cloud Style</label>
              <select
                value={settings.cloudStyle}
                onChange={(e) => setSettings({ ...settings, cloudStyle: e.target.value })}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-white hover:border-gray-600 focus:border-cyan-400 outline-none"
              >
                <option value="fluffy-whispy">Fluffy + Whispy</option>
                <option value="dramatic">Dramatic</option>
                <option value="clear-sky">Clear Sky</option>
              </select>
            </div>

            <div>
              <label className="block font-semibold mb-2">Twilight Style</label>
              <select
                value={settings.twilightStyle}
                onChange={(e) => setSettings({ ...settings, twilightStyle: e.target.value })}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-white hover:border-gray-600 focus:border-cyan-400 outline-none"
              >
                <option value="pink">Pink / Purple</option>
                <option value="golden">Golden Hour</option>
                <option value="blue">Blue Hour</option>
                <option value="orange">Orange Sunset</option>
              </select>
            </div>

            <div>
              <label className="block font-semibold mb-2">TV Screen Replacement</label>
              <select
                value={settings.tvScreenReplacement}
                onChange={(e) => setSettings({ ...settings, tvScreenReplacement: e.target.value })}
                className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-white hover:border-gray-600 focus:border-cyan-400 outline-none"
              >
                <option value="none">None</option>
                <option value="black">Black Screen</option>
                <option value="nature">Nature Scene</option>
                <option value="abstract">Abstract Art</option>
              </select>
            </div>
          </div>

          {/* Toggles */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
            {toggles.map((toggle) => (
              <div key={toggle.key} className="flex items-center justify-between py-2">
                <label className="font-medium">{toggle.label}</label>
                <button
                  onClick={() =>
                    setSettings({
                      ...settings,
                      [toggle.key]: !settings[toggle.key]
                    })
                  }
                  className={`relative inline-block w-14 h-7 rounded-full transition ${
                    settings[toggle.key] ? 'bg-cyan-400' : 'bg-gray-700'
                  }`}
                >
                  <div
                    className={`absolute top-1 w-5 h-5 bg-white rounded-full transition transform ${
                      settings[toggle.key] ? 'translate-x-7' : 'translate-x-1'
                    }`}
                  />
                </button>
              </div>
            ))}
          </div>

          {/* Save Buttons */}
          <div className="flex gap-4 pt-8 border-t border-gray-800">
            <button className="px-8 py-3 bg-cyan-400 text-black font-semibold rounded-lg hover:bg-cyan-300 transition">
              Save Changes
            </button>
            <button
              onClick={() => setSettings(defaultSettings)}
              className="px-8 py-3 border border-gray-700 text-white font-semibold rounded-lg hover:border-gray-600 transition"
            >
              Reset to Default
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Account Panel
function AccountPanel() {
  return (
    <div className="space-y-8">
      {/* Subscription Status */}
      <div className="bg-gray-950 border border-gray-800 rounded-xl p-8">
        <h2 className="text-2xl font-bold mb-6">Balance & Subscription</h2>
        <div className="space-y-4">
          <div className="flex justify-between items-center py-3 border-b border-gray-800">
            <span className="text-gray-400">Current Plan</span>
            <span className="text-white font-semibold">Free Tier</span>
          </div>
          <div className="flex justify-between items-center py-3 border-b border-gray-800">
            <span className="text-gray-400">Credits Remaining</span>
            <span className="text-3xl font-bold text-cyan-400">10</span>
          </div>
          <div className="flex justify-between items-center py-3 border-b border-gray-800">
            <span className="text-gray-400">Credits Used This Month</span>
            <span className="text-white">0</span>
          </div>
          <div className="flex justify-between items-center py-3">
            <span className="text-gray-400">Next Renewal</span>
            <span className="text-white">-</span>
          </div>
        </div>

        <div className="flex gap-4 mt-8">
          <Link
            href="/pricing"
            className="px-6 py-3 bg-cyan-400 text-black font-semibold rounded-lg hover:bg-cyan-300 transition"
          >
            Upgrade Plan
          </Link>
          <button className="px-6 py-3 bg-gray-800 text-white rounded-lg hover:bg-gray-700 transition">
            View Invoices
          </button>
          <button className="px-6 py-3 bg-gray-800 text-white rounded-lg hover:bg-gray-700 transition">
            Credit Log
          </button>
        </div>
      </div>

      {/* Billing History */}
      <div className="bg-gray-950 border border-gray-800 rounded-xl p-8">
        <h3 className="text-xl font-bold mb-4">Billing History</h3>
        <p className="text-gray-400">No billing history yet</p>
      </div>

      {/* Account Info */}
      <div className="bg-gray-950 border border-gray-800 rounded-xl p-8">
        <h3 className="text-xl font-bold mb-6">Account Information</h3>
        <div className="space-y-4">
          <div>
            <label className="block text-gray-400 text-sm mb-2">Email</label>
            <input
              type="email"
              value="james@virul.co"
              readOnly
              className="w-full bg-gray-900 border border-gray-800 rounded-lg px-4 py-3 text-white"
            />
          </div>
          <div>
            <label className="block text-gray-400 text-sm mb-2">Name</label>
            <input
              type="text"
              placeholder="Your name"
              className="w-full bg-gray-900 border border-gray-800 rounded-lg px-4 py-3 text-white focus:border-cyan-400 outline-none"
            />
          </div>
        </div>
        <button className="mt-6 px-6 py-3 bg-gray-800 text-white rounded-lg hover:bg-gray-700 transition">
          Update Account
        </button>
      </div>
    </div>
  );
}

// Team Panel
function TeamPanel() {
  const [newEmail, setNewEmail] = useState('');

  return (
    <div className="space-y-8">
      <div className="bg-gray-950 border border-gray-800 rounded-xl p-8">
        <h2 className="text-2xl font-bold mb-2">Add Your Team</h2>
        <p className="text-gray-400 mb-8">
          Who else can upload photos and view shoots on your account?
        </p>

        {/* Primary Email */}
        <div className="mb-8">
          <h3 className="text-lg font-semibold mb-4">Your Email (Owner)</h3>
          <div className="flex items-center gap-4 bg-gray-900 rounded-lg px-4 py-3">
            <Mail className="w-5 h-5 text-cyan-400" />
            <span className="text-white flex-1">james@virul.co</span>
            <span className="bg-cyan-400 text-black px-3 py-1 rounded text-sm font-semibold">
              Owner
            </span>
          </div>
        </div>

        {/* Add Team Member */}
        <div>
          <h3 className="text-lg font-semibold mb-4">Additional Team Members</h3>
          <div className="flex gap-4 mb-4">
            <input
              type="email"
              placeholder="Enter email address"
              value={newEmail}
              onChange={(e) => setNewEmail(e.target.value)}
              className="flex-1 bg-gray-900 border border-gray-800 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-cyan-400"
            />
            <button className="px-6 py-3 bg-cyan-400 text-black font-semibold rounded-lg hover:bg-cyan-300 transition">
              Add Member
            </button>
          </div>
          <p className="text-gray-500 text-sm">No team members added yet</p>
        </div>
      </div>
    </div>
  );
}

// Automation Panel
function AutomationPanel() {
  return (
    <div className="bg-gray-950 border border-gray-800 rounded-xl p-8">
      <h2 className="text-2xl font-bold mb-2">Dropbox Automation</h2>
      <p className="text-gray-400 mb-8">Automatically process photos when you upload to Dropbox</p>

      {/* Steps */}
      <div className="space-y-4 mb-8">
        {[
          { step: 1, text: 'Create a folder called "AutoHDR" in your Dropbox' },
          { step: 2, text: 'Inside "AutoHDR", create a folder for each listing (e.g., "123 Main St")' },
          { step: 3, text: 'Create a subfolder called "01-RAW-Photos" for your brackets' },
          { step: 4, text: 'Upload your RAW photos - we\'ll automatically process them!' }
        ].map((item) => (
          <div key={item.step} className="flex gap-4 items-start">
            <div className="w-8 h-8 rounded-full bg-cyan-400 text-black flex items-center justify-center font-bold flex-shrink-0">
              {item.step}
            </div>
            <p className="text-white pt-1">{item.text}</p>
          </div>
        ))}
      </div>

      <div className="bg-cyan-400 bg-opacity-10 border border-cyan-400 rounded-lg px-4 py-3 mb-8">
        <p className="text-cyan-400">
          Any RAW photo uploaded to 01-RAW-Photos will automatically be detected, processed with
          your style settings, and saved to a "02-Edited" folder!
        </p>
      </div>

      <div className="flex gap-4">
        <button className="px-6 py-3 bg-cyan-400 text-black font-semibold rounded-lg hover:bg-cyan-300 transition flex items-center gap-2">
          <Cloud className="w-5 h-5" />
          Connect Dropbox
        </button>
        <button className="px-6 py-3 border border-gray-700 text-white rounded-lg hover:border-gray-600 transition">
          Watch Tutorial
        </button>
      </div>
    </div>
  );
}

// Affiliate Panel
function AffiliatePanel() {
  const referralCode = 'JAMES2026';

  const copyToClipboard = () => {
    navigator.clipboard.writeText(referralCode);
    alert('Referral code copied!');
  };

  return (
    <div className="space-y-8">
      <div className="bg-gray-950 border border-gray-800 rounded-xl p-8">
        <h2 className="text-2xl font-bold mb-2">Refer Friends & Earn 50%</h2>
        <p className="text-gray-400 mb-8">
          Earn 50% commission on every user you refer. They get 25% off their first month!
        </p>

        {/* Referral Code */}
        <div className="bg-gray-900 rounded-lg p-6 mb-8">
          <p className="text-gray-400 text-sm mb-2">Your Referral Code</p>
          <div className="flex items-center gap-4">
            <span className="text-3xl font-bold text-cyan-400">{referralCode}</span>
            <button
              onClick={copyToClipboard}
              className="px-4 py-2 bg-gray-800 text-white rounded-lg hover:bg-gray-700 transition flex items-center gap-2"
            >
              <Copy className="w-4 h-4" />
              Copy
            </button>
          </div>
        </div>

        {/* Share Link */}
        <div className="mb-8">
          <p className="text-gray-400 text-sm mb-2">Share Link</p>
          <div className="flex gap-4">
            <input
              type="text"
              value={`https://autohdr.app/ref/${referralCode}`}
              readOnly
              className="flex-1 bg-gray-900 border border-gray-800 rounded-lg px-4 py-3 text-white"
            />
            <button className="px-6 py-3 bg-cyan-400 text-black font-semibold rounded-lg hover:bg-cyan-300 transition">
              Copy Link
            </button>
          </div>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-gray-900 rounded-lg p-6">
            <p className="text-gray-400 text-sm mb-1">Users Referred</p>
            <p className="text-3xl font-bold text-white">0</p>
          </div>
          <div className="bg-gray-900 rounded-lg p-6">
            <p className="text-gray-400 text-sm mb-1">Active Subscriptions</p>
            <p className="text-3xl font-bold text-white">0</p>
          </div>
          <div className="bg-gray-900 rounded-lg p-6">
            <p className="text-gray-400 text-sm mb-1">Total Earned</p>
            <p className="text-3xl font-bold text-cyan-400">$0.00</p>
          </div>
        </div>
      </div>

      {/* Payout Settings */}
      <div className="bg-gray-950 border border-gray-800 rounded-xl p-8">
        <h3 className="text-xl font-bold mb-4">Payout Settings</h3>
        <p className="text-gray-400 mb-4">
          Connect your PayPal or Stripe account to receive payouts.
        </p>
        <button className="px-6 py-3 bg-gray-800 text-white rounded-lg hover:bg-gray-700 transition">
          Setup Payouts
        </button>
      </div>
    </div>
  );
}
