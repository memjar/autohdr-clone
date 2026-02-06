/**
 * Style Settings API Routes
 * Saves and retrieves user style preferences
 */

import { NextRequest, NextResponse } from 'next/server';
import { StyleSettings } from '@/lib/types';

// In-memory store for demo
const styleSettings = new Map<string, StyleSettings>();

const DEMO_USER_ID = 'demo-user';

// Default settings
const defaultSettings: Omit<StyleSettings, 'id' | 'userId' | 'createdAt' | 'updatedAt'> = {
  interiorExterior: 'interior',
  brightness: 0,
  contrast: 0,
  vibrance: 0,
  whiteBalance: 0,
  windowPull: 'natural',
  cloudStyle: 'fluffy-whispy',
  twilightStyle: 'pink',
  tvScreen: 'none',
  fireplace: false,
  declutter: true,
  interiorClouds: true,
  exteriorClouds: true,
  deduplication: false,
  walkthrough: false,
  retainSky: false,
  perspective: true,
  grassReplacement: false,
  signRemoval: false,
  filenames: 'default',
};

// Initialize demo settings
styleSettings.set(DEMO_USER_ID, {
  id: 'settings_demo',
  userId: DEMO_USER_ID,
  ...defaultSettings,
  createdAt: new Date(),
  updatedAt: new Date(),
});

// GET /api/settings - Get user's style settings
export async function GET(req: NextRequest) {
  const userId = req.headers.get('x-user-id') || DEMO_USER_ID;

  let settings = styleSettings.get(userId);

  // Create default settings if none exist
  if (!settings) {
    settings = {
      id: `settings_${userId}`,
      userId,
      ...defaultSettings,
      createdAt: new Date(),
      updatedAt: new Date(),
    };
    styleSettings.set(userId, settings);
  }

  return NextResponse.json({
    success: true,
    data: settings,
  });
}

// PUT /api/settings - Update style settings
export async function PUT(req: NextRequest) {
  const userId = req.headers.get('x-user-id') || DEMO_USER_ID;

  try {
    const body = await req.json();
    const updates = body as Partial<StyleSettings>;

    let settings = styleSettings.get(userId);

    if (!settings) {
      settings = {
        id: `settings_${userId}`,
        userId,
        ...defaultSettings,
        createdAt: new Date(),
        updatedAt: new Date(),
      };
    }

    // Validate slider values
    const sliderKeys = ['brightness', 'contrast', 'vibrance', 'whiteBalance'] as const;
    for (const key of sliderKeys) {
      if (updates[key] !== undefined) {
        const value = updates[key] as number;
        if (value < -2 || value > 2) {
          return NextResponse.json(
            { success: false, error: `${key} must be between -2 and 2` },
            { status: 400 }
          );
        }
      }
    }

    // Merge updates
    const updatedSettings: StyleSettings = {
      ...settings,
      ...updates,
      id: settings.id,
      userId: settings.userId,
      createdAt: settings.createdAt,
      updatedAt: new Date(),
    };

    styleSettings.set(userId, updatedSettings);

    return NextResponse.json({
      success: true,
      data: updatedSettings,
      message: 'Settings saved successfully',
    });
  } catch (error) {
    console.error('Settings error:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to update settings' },
      { status: 500 }
    );
  }
}

// POST /api/settings/reset - Reset to defaults
export async function POST(req: NextRequest) {
  const userId = req.headers.get('x-user-id') || DEMO_USER_ID;

  const existingSettings = styleSettings.get(userId);

  const resetSettings: StyleSettings = {
    id: existingSettings?.id || `settings_${userId}`,
    userId,
    ...defaultSettings,
    createdAt: existingSettings?.createdAt || new Date(),
    updatedAt: new Date(),
  };

  styleSettings.set(userId, resetSettings);

  return NextResponse.json({
    success: true,
    data: resetSettings,
    message: 'Settings reset to defaults',
  });
}
